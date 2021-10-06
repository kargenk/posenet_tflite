import argparse
import sys
import time
from typing import Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tflite_runtime.interpreter import Interpreter

from visualize import parse_output, draw_keypoints


def build_interpreter(model_path: str, num_threds: int = None) -> tuple[Interpreter, list, list]:
    """
    TensorFlow Liteのインタプリタを初期化してテンソル割り当て後，取得した入出力の情報とともに返す関数.

    Args:
        model_path (str): モデルのパス
        num_threds (int, optional): 実行スレッド数. Defaults to None.

    Returns:
        tuple[Interpreter, list, list]: インタプリタと入力テンソルの情報のリストと出力の情報のリスト
    """
    # tfliteのインタプリタを初期化してテンソルを割り当て
    interpreter = Interpreter(model_path=model_path, num_threads=num_threds)
    interpreter.allocate_tensors()

    # 入出力の情報を取得
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return interpreter, input_details, output_details


def preprocess_image(source: Union[str, np.ndarray], input_details: list) -> tuple[np.ndarray, Image.Image]:
    """
    画像のパスまたは画像を受け取って，モデルの入力に合うようにリサイズ後に次元を拡張し，必要であれば正規化も行なって返す関数.

    Args:
        source (Union[str, np.ndarray]): 画像のパスまたは画像
        input_details (list): tfliteのモデルの入力テンソル情報

    Returns:
        tuple[np.ndarray, Image.Image]: 前処理後の画像と読み込んだままの画像
    """
    # 入力テンソルの情報[N, H, W, C]から画像をリサイズ
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    if isinstance(source, str):
        img_resized = Image.open(source).resize((width, height))
    else:
        img_resized = cv2.resize(source, (width, height))

    # add N(batch) dim
    img_input = np.expand_dims(img_resized, axis=0)

    # テンソルのデータ型をチェックして[0, 1]に正規化
    is_float_model = (input_details[0]['dtype'] == np.float32)
    if is_float_model:
        img_input = (np.float32(img_input) - args.input_mean) / args.input_std

    return img_input, img_resized


def main(args):
    interpreter, input_details, output_details = build_interpreter(args.model_file, args.num_threads)

    delay = 1
    window_name = 'test'
    cap = cv2.VideoCapture('data/just_do_it.mp4')
    if not cap.isOpened():
        sys.exit()

    while True:
        ret, frame = cap.read()
        if ret:
            img_input, img_resized = preprocess_image(frame, input_details)
            interpreter.set_tensor(input_details[0]['index'], img_input)  # テンソルを入力

            # 実行時間を計測して出力
            start_time = time.time()
            interpreter.invoke()
            stop_time = time.time()
            print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))

            # モデルの出力は入力を9x9に分割した各ジョイントの確率マップと
            # より正確に位置を計算するためのキーポイントのx座標(最初の17次元)とy座標(後の17次元)
            heatmaps = interpreter.get_tensor(output_details[0]['index'])
            offset_vectors = interpreter.get_tensor(output_details[1]['index'])
            heatmaps, offsets = np.squeeze(heatmaps), np.squeeze(offset_vectors)

            keypoints = parse_output(heatmaps, offsets, 0)
            # print(keypoints)

            img = draw_keypoints(np.asarray(img_resized), keypoints)

            cv2.imshow(window_name, img)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cv2.destroyWindow(window_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--image',
        default='sample/grace_hopper.bmp',
        help='image to be classified')
    parser.add_argument(
        '-m',
        '--model_file',
        default='models/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite',
        help='.tflite model to be executed')
    parser.add_argument(
        '--input_mean',
        default=127.5, type=float,
        help='input mean')
    parser.add_argument(
        '--input_std',
        default=127.5, type=float,
        help='input standard deviation')
    parser.add_argument(
        '--num_threads', default=None, type=int, help='number of threads')
    args = parser.parse_args()

    main(args)