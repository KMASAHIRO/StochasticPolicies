from TrafficFlowControl_with_StochasticPolicies.train.train_gym import train_agent_gym
import argparse
import logging

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, required=True)
    parser.add_argument("--model_save_path", type=str, required=True)
    parser.add_argument("--episode_per_learn", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=1400)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--decay_rate", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--layer_type", type=str, default="")
    parser.add_argument("--bbb_pi", type=float, default=0.5)
    parser.add_argument("--bbb_sigma1", type=float, default=-0)
    parser.add_argument("--bbb_sigma2", type=float, default=-6)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--log_dir", type=str, default="./")
    parser.add_argument("--learn_curve_csv", type=str, default="")
    parser.add_argument("--device", type=str, default="cpu")
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


    if args.learn_curve_csv == "":
        learn_curve_csv = None
    else:
        learn_curve_csv = args.learn_curve_csv

    if args.layer_type == "":
        layer_type = None
    else:
        layer_type = args.layer_type

    try:
        train_agent_gym(
            env_name=args.env_name, model_save_path=args.model_save_path, 
            episode_per_learn=args.episode_per_learn, episodes=args.episodes, max_steps=args.max_steps, 
            lr=args.lr, decay_rate=args.decay_rate, temperature=args.temperature, noise=args.noise, 
            layer_type=layer_type, bbb_pi=args.bbb_pi, bbb_sigma1=args.bbb_sigma1, bbb_sigma2=args.bbb_sigma2, 
            gamma=args.gamma, 
            log_dir=args.log_dir, learn_curve_csv=learn_curve_csv,
            device=args.device
            )
    except Exception as err:
        logger.exception("The program stopped because of this error.")