import numpy as np
import os
import cv2
import json

import matplotlib
from tqdm import tqdm


def create_output_video(args, env, output_filename):
    output_video_filename = os.path.splitext(output_filename)[0] + (f'_{args.speedup}x.avi' if args.speedup else '.avi')

    fourcc = cv2.VideoWriter_fourcc('M', 'P', 'E', 'G')
    input_video = cv2.VideoCapture(env.input_video_filename)
    fps = input_video.get(5)
    width = int(input_video.get(3))
    height = int(input_video.get(4))
    output_video = cv2.VideoWriter(output_video_filename, fourcc, fps, (width, height))

    print('\nCreating output video at {}'.format(os.path.abspath(output_video_filename)))
    skips = list(np.hstack([[0], np.array(env.selected_frames)[1:] - np.array(env.selected_frames)[:-1]]))
    input_video.set(1, env.selected_frames[0])
    check, frame = input_video.read()
    if args.annotations_filename is not None:
        start_point = (0, 0)
        end_point = (width, height)
        colors = ['green', 'blue', 'red', 'yellow', 'black', 'firebrick', 'darkgoldenrod', 'forestgreen', 'steelblue', 'indianred', 'mediumpurple', 'brown', 'cyan', 'magenta']
        color_id = 0
        region_id = 0
        if args.dataset == 'YouCook2':
            video_annotations = json.load(open(args.annotations_filename))['database'][env.experiment_name]['annotations']
        elif args.dataset == 'COIN':
            video_annotations = json.load(open(args.annotations_filename))['database'][env.experiment_name]['annotation']

    frame_idx = 0
    for idx, skip in enumerate(tqdm(skips)):
        for _ in range(skip):
            check, frame = input_video.read()
            frame_idx += 1

        if check:
            if args.print_details:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, '{}x'.format(skips[idx]), (50, 100), font, round(height/240), (255, 255, 255), height//120, cv2.LINE_AA)

                if region_id < len(video_annotations) and args.annotations_filename is not None:
                    if (frame_idx >= video_annotations[region_id]['segment'][0] * fps):
                        frame = cv2.rectangle(frame, start_point, end_point, tuple([int(i*255) for i in matplotlib.colors.to_rgb(colors[color_id])]), int(height/30))

                    if (frame_idx >= video_annotations[region_id]['segment'][1]*fps):  # If previous segment has ended. Let's go to the next one
                        region_id += 1
                        color_id += 1

            output_video.write(frame)
    print('\nDone!')

    input_video.release()
    output_video.release()