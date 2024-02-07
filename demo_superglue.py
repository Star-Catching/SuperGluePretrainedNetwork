#! /usr/bin/env python3
#
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#                       Daniel DeTone
#                       Tomasz Malisiewicz
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from pathlib import Path
import argparse
import cv2
import matplotlib.cm as cm
import torch

from models.matching import Matching
from models.utils import (AverageTimer, VideoStreamer,
                          make_matching_plot_fast, frame2tensor)

torch.set_grad_enabled(False)


if __name__ == '__main__':
    # 创建 ArgumentParser 对象，用于解析命令行参数
    parser = argparse.ArgumentParser(
        description='SuperGlue demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # 添加命令行参数

    # 输入源，可以是USB摄像头ID、IP摄像头URL，或者图像目录或视频文件的路径
    parser.add_argument(
        '--input', type=str, default='0',
        help='ID of a USB webcam, URL of an IP camera, '
             'or path to an image directory or movie file')
    # 输出帧的目录，如果为None，则不输出
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Directory where to write output frames (If None, no output)')

    # 图像文件的通配符，如果指定了图像目录
    parser.add_argument(
        '--image_glob', type=str, nargs='+', default=['*.png', '*.jpg', '*.jpeg'],
        help='Glob if a directory of images is specified')
    # 如果输入为视频或图像目录，跳过的图像数量
    parser.add_argument(
        '--skip', type=int, default=1,
        help='Images to skip if input is a movie or directory')
    # 如果输入为视频或图像目录，指定的最大处理长度
    parser.add_argument(
        '--max_length', type=int, default=1000000,
        help='Maximum length if input is a movie or directory')
    # 调整输入图像大小，可以指定为两个数字，也可以指定一个数字，-1 表示不调整大小
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
    
    # SuperGlue 模型的权重，可以选择 'indoor' 或 'outdoor'
    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='indoor',
        help='SuperGlue weights')
    # SuperPoint 检测的最大关键点数量，-1 表示保留所有关键点
    parser.add_argument(
        '--max_keypoints', type=int, default=-1,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    # SuperPoint 关键点检测的置信度阈值
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    # SuperPoint 非最大抑制半径，必须是正数
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
        ' (Must be positive)')
    # SuperGlue 中 Sinkhorn 迭代的次数
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    # SuperGlue 匹配的阈值
    parser.add_argument(
        '--match_threshold', type=float, default=0.2,
        help='SuperGlue match threshold')

    # 是否显示检测到的关键点
    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Show the detected keypoints')
    # 是否禁止在屏幕上显示图像，对于远程运行时很有用
    parser.add_argument(
        '--no_display', action='store_true',
        help='Do not display images to screen. Useful if running remotely')
    # 是否强制使用CPU模式
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')

    opt = parser.parse_args()
    print(opt)

    # 如果 resize 参数包含两个值且第二个值为 -1
    if len(opt.resize) == 2 and opt.resize[1] == -1:
        # 将 resize 参数截取为包含第一个值的列表
        opt.resize = opt.resize[0:1]
    # 根据不同情况打印将要进行的图像大小调整操作
    if len(opt.resize) == 2:
        # 如果 resize 参数包含两个值，打印将要调整的宽度和高度
        print('Will resize to {}x{} (WxH)'.format(
            opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        # 如果 resize 参数只包含一个正整数，打印将要调整的最大尺寸
        print('Will resize max dimension to {}'.format(opt.resize[0]))
    elif len(opt.resize) == 1:
        # 如果 resize 参数只包含一个值，且为 -1，表示不进行大小调整
        print('Will not resize images')
    else:
        # 如果 resize 参数包含超过两个值，抛出数值错误
        raise ValueError('Cannot specify more than two integers for --resize')

    # 判断是否有可用的 GPU，并且用户没有强制使用 CPU 模式
    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    # 创建配置字典 config，包含了 SuperPoint 和 SuperGlue 模型的相关参数
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,                   # SuperPoint 非最大抑制半径
            'keypoint_threshold': opt.keypoint_threshold,   # SuperPoint 关键点检测的置信度阈值
            'max_keypoints': opt.max_keypoints              # SuperPoint 检测的最大关键点数量
        },
        'superglue': {
            'weights': opt.superglue,                        # SuperGlue 模型的权重（'indoor' 或 'outdoor'）
            'sinkhorn_iterations': opt.sinkhorn_iterations,   # SuperGlue 中 Sinkhorn 迭代的次数
            'match_threshold': opt.match_threshold,            # SuperGlue 匹配的阈值
        }
    }
    # 创建匹配模型实例，并将其设置为评估模式，然后移动到指定设备
    # * Matching(config)：创建了一个名为Matching的类的实例，其中传入了参数config。这个类可能是一个包含图像匹配功能的模型类。
    # * .eval()：将创建的模型实例设置为评估模式。在评估模式下，模型会禁用一些具有随机性质的操作，如Dropout。这是因为在评估模式下，我们不需要模型进行随机的行为，而是希望其输出稳定且可重复。
    # * .to(device)：将模型移动到指定的设备上，其中device是一个表示硬件设备的对象，比如GPU或CPU。这个步骤是为了利用硬件加速，提高模型的运行速度
    matching = Matching(config).eval().to(device)
    keys = ['keypoints', 'scores', 'descriptors'] # 定义包含 SuperPoint 模型输出关键信息的列表

    # 创建 VideoStreamer 对象，处理视频流输入
    vs = VideoStreamer(opt.input, opt.resize, opt.skip,
                       opt.image_glob, opt.max_length)
    # 这段代码从 vs（VideoStreamer 对象）获取下一帧的图像。
    # 它使用 next_frame() 方法，并返回帧图像 frame 和一个布尔值 ret，表示是否成功读取帧。
    # 接着，使用 assert 语句检查是否成功读取第一帧，如果未成功，抛出 AssertionError，并输出错误消息。
    # 获取下一帧图像
    frame, ret = vs.next_frame()
    # 使用 assert 语句检查是否成功读取第一帧
    assert ret, 'Error when reading the first frame (try different --input?)'

    # 这段代码对获取的图像帧进行处理，将其转换为张量，并使用 SuperPoint 模型进行关键点检测。
    # 然后，将得到的关键点信息存储在 last_data 中，将帧张量、图像和帧编号等信息保存在对应的变量中。
    frame_tensor = frame2tensor(frame, device)   # 将获取的图像帧转换为张量
    last_data = matching.superpoint({'image': frame_tensor})   # 使用 SuperPoint 模型进行关键点检测
    # print(last_data)
    last_data = {k+'0': last_data[k] for k in keys}   # 选择关键信息存储在 last_data 中，包括关键点、分数和描述符
    last_data['image0'] = frame_tensor  # 将图像张量和相关信息存储在 last_data 中
    last_frame = frame  # 存储最后一帧的图像
    last_image_id = 0  # 存储最后一帧的图像编号

    # 这段代码检查是否指定了输出目录 opt.output_dir，如果有的话，它会输出一条提示信息并创建该目录（如果目录不存在）
    if opt.output_dir is not None:
        print('==> Will write outputs to {}'.format(opt.output_dir))
        Path(opt.output_dir).mkdir(exist_ok=True)

    # Create a window to display the demo.
    # 这段代码创建一个窗口以显示演示结果。
    # 如果 opt.no_display 为 False（即不禁用显示），则使用 OpenCV 创建一个窗口，并设置窗口的名称为 'SuperGlue matches'。
    # 同时，设置窗口的大小为640*2宽度和480高度。如果 opt.no_display 为 True，则输出一条消息表示跳过可视化，不会显示 GUI 界面。
    if not opt.no_display:
        cv2.namedWindow('SuperGlue matches', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('SuperGlue matches', 640*2, 480)
    else:
        print('Skipping visualization, will not show a GUI.')

    # Print the keyboard help menu.
    print('==> Keyboard control:\n'
          '\tn: select the current frame as the anchor\n'
          '\te/r: increase/decrease the keypoint confidence threshold\n'
          '\td/f: increase/decrease the match filtering threshold\n'
          '\tk: toggle the visualization of keypoints\n'
          '\tq: quit')

    # 创建一个平均计时器对象
    timer = AverageTimer()
    # 进入无限循环
    while True:
        # 获取视频流的下一帧图像和读取状态
        frame, ret = vs.next_frame()
        # 如果读取失败，输出消息表示演示结束，并退出循环
        if not ret:
            print('Finished demo_superglue.py')
            break
        timer.update('data')
        stem0, stem1 = last_image_id, vs.i - 1
        # 将当前帧图像转换为张量
        frame_tensor = frame2tensor(frame, device)
        # 使用匹配模型进行匹配
        pred = matching({**last_data, 'image1': frame_tensor})
        # 提取关键点坐标、匹配信息和匹配置信度
        kpts0 = last_data['keypoints0'][0].cpu().numpy()
        kpts1 = pred['keypoints1'][0].cpu().numpy()
        matches = pred['matches0'][0].cpu().numpy()
        confidence = pred['matching_scores0'][0].cpu().numpy()
        # 更新计时器的 'forward' 部分
        timer.update('forward')

        # 过滤出有效的匹配（matches > -1）
        valid = matches > -1

        # 提取有效匹配对应的关键点坐标
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        # 为有效匹配生成相应的颜色（根据置信度）
        color = cm.jet(confidence[valid])
        # 这段代码创建了一个包含文本信息的列表 text，其中包括 'SuperGlue'、关键点数量的信息以及匹配数量的信息。
        text = [
            'SuperGlue',  # 模型名称
            'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),  # 关键点数量信息
            'Matches: {}'.format(len(mkpts0))   # 匹配数量信息
        ]
        # 这段代码获取了匹配模型 matching 中 SuperPoint 部分的关键点阈值 k_thresh 和 SuperGlue 部分的匹配阈值 m_thresh。
        # 然后，创建了一个包含小文本信息的列表 small_text，其中包括关键点阈值、匹配阈值以及图像对的信息。
        # 获取 SuperPoint 模型中的关键点阈值和 SuperGlue 模型中的匹配阈值
        k_thresh = matching.superpoint.config['keypoint_threshold']
        m_thresh = matching.superglue.config['match_threshold']

        # 创建包含小文本信息的列表
        small_text = [
            'Keypoint Threshold: {:.4f}'.format(k_thresh),  # 关键点阈值信息
            'Match Threshold: {:.2f}'.format(m_thresh),      # 匹配阈值信息
            'Image Pair: {:06}:{:06}'.format(stem0, stem1),  # 图像对的信息
        ]
        # 调用 make_matching_plot_fast 函数生成匹配结果的可视化图
        out = make_matching_plot_fast(
            last_frame, frame,   # 前一帧和当前帧的图像
            kpts0, kpts1,        # 关键点坐标
            mkpts0, mkpts1,      # 有效匹配的关键点坐标
            color,               # 匹配的颜色信息
            text,                # 主要文本信息
            path=None,           # 图像保存路径（未指定）
            show_keypoints=opt.show_keypoints,  # 是否显示关键点
            small_text=small_text  # 小文本信息
            )
        # 如果不禁用显示
        if not opt.no_display:
            # 使用 OpenCV 显示生成的匹配可视化图
            cv2.imshow('SuperGlue matches', out)

            # 检测键盘按键
            key = chr(cv2.waitKey(1) & 0xFF)

            # 根据按键执行相应的操作
            if key == 'q':
                # 退出程序（按 'q' 键）
                vs.cleanup()
                print('Exiting (via q) demo_superglue.py')
                break
            elif key == 'n':  # set the current frame as anchor 将当前帧设为锚点（按 'n' 键）
                last_data = {k+'0': pred[k+'1'] for k in keys}
                last_data['image0'] = frame_tensor
                last_frame = frame
                last_image_id = (vs.i - 1)
            elif key in ['e', 'r']:
                # Increase/decrease keypoint threshold by 10% each keypress.  # 增加/减少关键点阈值 10%（按 'e' 或 'r' 键）
                d = 0.1 * (-1 if key == 'e' else 1)
                matching.superpoint.config['keypoint_threshold'] = min(max(
                    0.0001, matching.superpoint.config['keypoint_threshold']*(1+d)), 1)
                print('\nChanged the keypoint threshold to {:.4f}'.format(
                    matching.superpoint.config['keypoint_threshold']))
            elif key in ['d', 'f']:
                # Increase/decrease match threshold by 0.05 each keypress. 增加/减少匹配阈值 0.05（按 'd' 或 'f' 键）
                d = 0.05 * (-1 if key == 'd' else 1)
                matching.superglue.config['match_threshold'] = min(max(
                    0.05, matching.superglue.config['match_threshold']+d), .95)
                print('\nChanged the match threshold to {:.2f}'.format(
                    matching.superglue.config['match_threshold']))
            elif key == 'k':                    # 切换显示关键点的状态（按 'k' 键）
                opt.show_keypoints = not opt.show_keypoints

        timer.update('viz')
        timer.print()
        # 如果指定了输出目录
        if opt.output_dir is not None:
            #stem = 'matches_{:06}_{:06}'.format(last_image_id, vs.i-1)  # 生成文件名（stem）
            stem = 'matches_{:06}_{:06}'.format(stem0, stem1)
            # 拼接输出文件的完整路径
            out_file = str(Path(opt.output_dir, stem + '.png'))
            # 打印输出文件信息
            print('\nWriting image to {}'.format(out_file))
             # 保存匹配可视化图为图片文件
            cv2.imwrite(out_file, out)
    # 关闭所有 OpenCV 窗口
    cv2.destroyAllWindows()
    # 清理 VideoStreamer
    vs.cleanup()
