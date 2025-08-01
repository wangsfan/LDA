import rosbag      # pip install bagpy


from tqdm import tqdm
import os
import numpy as np
import pickle
import torch

import sys
import events_to_frames

sys.path.append('/home/yjy/EventTransformerPlus-main/EventTransformerPlus-main'); import transformations


"""
1. DEPTHS: Store as single images
2: EVENTS: Store N images of length DEPTH/chunks_per_depth. Default: chunks_per_depth = 1
3: GRAYSCALE IMAGES: store the closest ones to Depth
"""

def stamp_to_us(stamp): return stamp.secs*1000000 + stamp.nsecs//1000

min_patches = 3

# Define source data
path_dataset = '/home/yjy/DATA/MVSEC/'
# sequence = 'outdoor_day/outdoor_day1'
# sequence = 'outdoor_day/outdoor_day2'
# sequence = 'outdoor_night/outdoor_night1'
# sequence = 'outdoor_night/outdoor_night2'
# sequence = 'outdoor_night/outdoor_night3'
for sequence in [
        'outdoor_day1',
        # 'outdoor_day/outdoor_day2',
        # 'outdoor_night1',
        # 'outdoor_night/outdoor_night2',
        # 'outdoor_night/outdoor_night3',
        ]:
    side = 'left'  
    
    
    ts_depths_ms = 50
    ts_depths_us = 50*1000
    
    
    avg_img_us = (1000/45)*1000 if 'day' in sequence else (1000/10)*1000
    height, width = 260, 346
    
    chunks_per_depth, k, minTime, maxTime = 1,  3, -1, 256*1000
    if minTime == -1: minTime = ts_depths_ms*1000 / k
    
    # store_depth, store_events, store_images = True, True, True
    store_depth, store_events, store_images = True, True, True
    # store_depth, store_events, store_images = False, False, True
    
    
    # Create destination folders    
    path_dataset_dst =  '/home/yjy/datasets/MVSEC_final/'
    
    store_folder = path_dataset_dst + f'0601_dataset_frames_k{k}_MT{int(maxTime//1000)}_mT{int(minTime//1000)}_v0_clear/{sequence.split("/")[-1]}/{side}/'
    
    
    
    if not os.path.isdir(store_folder): os.makedirs(store_folder)
    folder_event = store_folder + 'event_frames/' 
    folder_image_raw = store_folder + 'image_raw/'
    folder_depth = store_folder + 'depth/'
    if not os.path.isdir(folder_event): os.makedirs(folder_event)
    if not os.path.isdir(folder_image_raw): os.makedirs(folder_image_raw)
    if not os.path.isdir(folder_depth): os.makedirs(folder_depth)
    
    
    
    """
    1. Get depth
    2. Iterate for events, image
    """
    
    data = rosbag.Bag(path_dataset + sequence + '_data.bag', 'r')
    gt = rosbag.Bag(path_dataset + sequence + '_gt.bag', 'r')
    print(' * RosBags created')
    
    
    # %%
    
    # Initialize data loaders
    events_gen = data.read_messages(topics=[f'/davis/{side}/events'])
    if store_images: image_gen = data.read_messages(topics=[f'/davis/{side}/image_raw'])
    depth_gen = gt.read_messages(topics=[f'/davis/{side}/depth_image_raw'])
    depth_count = gt.get_message_count(topic_filters=[f'/davis/{side}/depth_image_raw'])
    
    print(' * Data generators initialized')
    
    # Initialize data buffers
    buffer_events = np.empty((0,4), dtype=np.uint)
    buffer_images = np.empty((0,height,width), dtype=np.uint8)
    buffer_images_ts = np.empty((0,), dtype=np.uint)
    
    # Initialize FIFOs
    pos_fifo = np.full((height, width, k), -np.inf, dtype=np.float32)
    neg_fifo = np.full((height, width, k), -np.inf, dtype=np.float32)
    
    pbar = tqdm(total=depth_count)
    prev_ts = -np.inf
    num_skipped = 0
    eventsnums = 0
    for num_iter, (_, depth, _) in enumerate(depth_gen):
        pbar.update(1)
        depthnum = 0
        try:
            
        
            ts = stamp_to_us(depth.header.stamp)
            ts_img_min, ts_img_max = prev_ts+avg_img_us, ts+avg_img_us
            depth = np.frombuffer(depth.data, dtype=np.float32)
            depth = depth.reshape(height, width)
        
            # Get events
            for _, events, _  in events_gen:
                events = np.array([ (e.x, e.y, stamp_to_us(e.ts), int(e.polarity)) \
                                 for e in events.events ]).astype(np.uint)
                if eventsnums == 0:
                    innitts = events[0,2].astype(np.float64)
                    eventsnums = 1
                if depthnum == 0:
                    depthnum = 1
                    ts = ts - innitts
                    ts_img_max = ts_img_max - innitts
                
                events[:,2] = events[:,2] - innitts
                buffer_events = np.append(buffer_events, events, axis=0)
                if events[:,2].max() > (ts): break
            # Remove old events. Useful at the beginning of the stream where depth maps are not yet being created
            buffer_events = buffer_events[buffer_events[:,2] > buffer_events[:,2].max()-maxTime]
            iter_events_inds = buffer_events[:,2] <= ts
            iter_events = buffer_events[iter_events_inds]
            buffer_events = buffer_events[~iter_events_inds]
            min_event_ts, max_event_ts = iter_events[:,2].min(), iter_events[:,2].max()
            diff_event_ts = ts - min_event_ts
            # Split events and generate representations. Create chunks_per_depth frames per depth map. time-window of ~50ms
            if diff_event_ts < ts_depths_us*1.3: 
                chunk_len_us = diff_event_ts/chunks_per_depth
                chunk_len_us += 1
            else:
                print('***', num_iter, '*** Many event representations...')
                chunk_len_us = ts_depths_us / chunks_per_depth
                
            # Generate event frames from events
            (iter_events_repr, min_max_values), (pos_fifo, neg_fifo) = events_to_frames.process_event_stream(iter_events, height, width, 
                                                                                 chunk_len_us, k, minTime, maxTime, pos_fifo, neg_fifo)
            assert iter_events_repr.shape[0] == 1 or num_iter == 0
            
            # break
        
            # Convert to patches
            events_pad = transformations.pad(torch.tensor(iter_events_repr[None,].reshape(1, iter_events_repr.shape[0], height, width, k*2).copy()), patch_size=12, pad_value=0.0)
            patches, pixels = transformations.window_partition( events_pad, patch_size=12, validation=False,
                           min_activations_per_patch=7.5, 
                           drop_token=0.0, 
                            chunk_len_ms=ts_depths_ms, maxTime=maxTime, 
                            patch_by_last_k=True,
                            reduce_tokens=True)   
            # activated_patches = patches.sum(-1).sum(-1) == 0
            num_patches = (patches.sum(-1) != 0).sum(-1)
            
            iter_events_repr = iter_events_repr.astype(np.float16)
            
            # Get grayscale imagesssss
            if len(buffer_images_ts) == 0 or buffer_images_ts[-1] < ts_img_max:
                for _, image, _ in image_gen:   # pass
                    ts_image = stamp_to_us(image.header.stamp) - innitts
                    image = np.frombuffer(image.data, dtype=np.uint8)
                    image = image.reshape(height, width)
                    buffer_images = np.append(buffer_images, [image], axis=0)
                    buffer_images_ts = np.append(buffer_images_ts, ts_image)
                    if ts_image > ts_img_max: break
            iter_images = buffer_images[buffer_images_ts <= ts_img_max]
            iter_images_ts = buffer_images_ts[buffer_images_ts <= ts_img_max]
            buffer_images = buffer_images[buffer_images_ts > ts_img_max]
            buffer_images_ts = buffer_images_ts[buffer_images_ts > ts_img_max]
            
            
            # if iter_events_repr
            
            
            filename = f'ts_{ts}.pckl'
            
            # Store depth maps, images and frames
            if len(iter_events_repr) <= chunks_per_depth:
                if num_patches[0][0] >= min_patches:
                    if store_depth: pickle.dump(depth, open(folder_depth + filename, 'wb'))
                    if store_events: pickle.dump(iter_events_repr, open(folder_event + filename, 'wb'))
                    if store_images: pickle.dump(iter_images, open(folder_image_raw + filename, 'wb'))
                else:
                    num_skipped += 1
                    print(f' - Skipping ts: {num_iter} ||| {ts} ||| num_patches {num_patches} ||| skipped {num_skipped}')
            else:
                if store_depth: pickle.dump(depth, open(folder_depth + filename, 'wb'))

                # Multiple event frames for a single depth map 
                min_event_ts, max_event_ts = iter_events[:,2].min(), iter_events[:,2].max()
                iter_events_ts_loop = np.arange(int(max_event_ts), int(min_event_ts), -chunk_len_us*chunks_per_depth)[::-1]
                iter_events_ts = np.arange(int(max_event_ts), int(min_event_ts), -chunk_len_us)[:len(iter_events_repr)][::-1]
                
                print(chunk_len_us, diff_event_ts, len(iter_events_repr), 
                      min_event_ts, max_event_ts, chunks_per_depth, len(iter_events_ts_loop), 
                      '||', ts_depths_ms, ts_depths_us, minTime)
                for i in range(len(iter_events_ts_loop)):
                    iter_ts = iter_events_ts_loop[i]
                    iter_prev_ts = iter_events_ts_loop[i-1] if i != 0 else -np.inf
                    
                    init_iter_events_repr = iter_events_repr[np.where((iter_events_ts > iter_prev_ts) & (iter_events_ts <= iter_ts))[0]]
                    init_iter_images = iter_images[np.where((iter_images_ts > iter_prev_ts) & (iter_images_ts <= iter_ts))[0]]
        
                    print(f'\t{i:03}/{len(iter_events_ts_loop):03} | {num_iter+1:05}/{depth_count:05} | time-window: {(iter_ts - iter_prev_ts)/1000:.2f} ms | ' +
                          f'{len(init_iter_events_repr)} events repr. | '+
                          f'{len(init_iter_images)} images | '
                          )
                    
                    filename = f'ts_{int(iter_ts)}.pckl'
                    if num_patches[0][i] >= min_patches:
                        if store_events: pickle.dump(init_iter_events_repr, open(folder_event + filename, 'wb'))
                        if store_images: pickle.dump(init_iter_images, open(folder_image_raw + filename, 'wb'))
                    else:
                        print(f' - Skipping ts: {num_iter} {i} {iter_ts} ||| num_patches {num_patches[0][i]} ||| skipped {num_skipped}')

            prev_ts = ts
        
        except Exception as e:
            print(1)

    print(store_folder)        
    print(sequence)
    







