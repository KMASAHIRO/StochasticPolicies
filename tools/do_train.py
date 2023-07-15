from ast import parse
from TrafficFlowControl_with_StochasticPolicies.train.train import train_PPO
import subprocess
import argparse
import logging

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--map_name", type=str, required=True)
    parser.add_argument("--map_dir", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=1400)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--layer_type", type=str, default="")
    parser.add_argument("--update_interval", type=int, default=1024)
    parser.add_argument("--minibatch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--entropy_coef", type=float, default=0.001)
    parser.add_argument("--bbb_pi", type=float, default=0.5)
    parser.add_argument("--bbb_sigma1", type=float, default=-0)
    parser.add_argument("--bbb_sigma2", type=float, default=-6)
    parser.add_argument("--log_dir", type=str, default="./")
    parser.add_argument("--reward_csv", type=str, default="")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--port", type=int, default=-1)
    parser.add_argument("--libsumo", action="store_true")
    parser.add_argument("-e", "--error_output_path", type=str, default="error_message.log")
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    # ファイル出力ハンドラーの設定
    handler = logging.FileHandler(args.error_output_path)
    handler.setLevel(logging.DEBUG)
    # 出力フォーマットの設定
    formatter = logging.Formatter('%(levelname)s  %(asctime)s  [%(name)s] %(message)s')
    handler.setFormatter(formatter)
    # ハンドラーの追加
    logger.addHandler(handler)
    # 重複出力をなくす
    logger.propagate = False

    if args.reward_csv == "":
        reward_csv = None
    else:
        reward_csv = args.reward_csv

    if args.layer_type == "":
        layer_type = None
    else:
        layer_type = args.layer_type
        
    if args.port == -1:
        port = None
    else:
        port = args.port
    
    try:
        train_PPO(
            run_name=args.run_name, map_name=args.map_name, map_dir=args.map_dir,  
            episodes=args.episodes, 
            temperature=args.temperature, noise=args.noise, layer_type=layer_type,  
            update_interval=args.update_interval, minibatch_size=args.minibatch_size, epochs=args.epochs, 
            entropy_coef = args.entropy_coef, 
            bbb_pi=args.bbb_pi, bbb_sigma1=args.bbb_sigma1, bbb_sigma2=args.bbb_sigma2, 
            log_dir=args.log_dir, reward_csv=reward_csv, 
            device=args.device, port=port, libsumo=args.libsumo, 
            )
    except Exception as err:
        logger.exception("The program stopped because of this error.")
