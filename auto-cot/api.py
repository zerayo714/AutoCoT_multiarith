import argparse
from utils import *


def cot(method, question):
    args = parse_arguments()
    decoder = Decoder()

    args.method = method
    if args.method != "zero_shot_cot":  
        if args.method == "auto_cot":  #auto_cot
            args.demo_path = "demos/multiarith_auto"   #前面訓練產生的dict 作為下一次迭代的對象
        else:
            args.demo_path = "demos/multiarith_manual" # manual_cot
        demo = create_demo_text(args, cot_flag=True)
    else:   #Zero_shot 跟 zero_shot_cot
        demo = None

    x = "Q: " + question + "\n" + "A:"      #x --> output架構 
    print('*****************************')
    print("Test Question:")
    print(question)
    print('*****************************')

    if args.method == "zero_shot":              #The answer is: 
        x = x + " " + args.direct_answer_trigger_for_zeroshot 
    elif args.method == "zero_shot_cot":        #Let's think step by step
        x = x + " " + args.cot_trigger 
    elif args.method == "manual_cot":           #不用trigger
        x = demo + x    
    elif args.method == "auto_cot":             #Let's think step by step
        x = demo + x + " " + args.cot_trigger   # The answer is \n Let's think step by step
    else:
        raise ValueError("method is not properly defined ...")

    print("Prompted Input:")
    print(x.replace("\n\n", "\n").strip())
    print('*****************************')

    max_length = args.max_length_cot if "cot" in args.method else args.max_length_direct  #限制CoT token數量(default=)
    z = decoder.decode(args, x, max_length)   #轉成vector
    z = z.replace("\n\n", "\n").replace("\n", "").strip()
    if args.method == "zero_shot_cot":
        # print(z)      CoT
        z2 = x + z + " " + args.direct_answer_trigger_for_zeroshot_cot
        # print(z2)     Q:...A:Let's...The answer is
        max_length = args.max_length_direct
        pred = decoder.decode(args, z2, max_length)
        #print(pred)    答案
        print("Output:")
        print(z  + "\n" + args.direct_answer_trigger_for_zeroshot_cot + pred)
        print('*****************************')
    else:
        # print(z)
        pred = z
        print("Output:")
        print(pred)
        print('*****************************')

def parse_arguments():
    parser = argparse.ArgumentParser(description="Zero-shot-CoT")

    parser.add_argument("--max_num_worker", type=int, default=0, help="maximum number of workers for dataloader")
    parser.add_argument(
        "--model", type=str, default="gpt3-xl", help="model used for decoding. Note that 'gpt3' are the smallest models."
    )
    parser.add_argument(
        "--method", type=str, default="auto_cot", choices=["zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot", "auto_cot"], help="method"
    )
    parser.add_argument(
        "--cot_trigger_no", type=int, default=1, help="A trigger sentence that elicits a model to execute chain of thought"
    )
    parser.add_argument(
        "--max_length_cot", type=int, default=256, help="maximum length of output tokens by model for reasoning extraction"
    )
    parser.add_argument(
        "--max_length_direct", type=int, default=32, help="maximum length of output tokens by model for answer extraction"
    )
    parser.add_argument(
        "--limit_dataset_size", type=int, default=0, help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    )
    parser.add_argument(
        "--api_time_interval", type=float, default=1.0, help=""
    )
    parser.add_argument(
        "--temperature", type=float, default=0, help=""
    )
    parser.add_argument(
        "--log_dir", type=str, default="./log/", help="log directory"
    )
    args = parser.parse_args()
    
    args.direct_answer_trigger_for_fewshot = "The answer is"
    args.direct_answer_trigger_for_zeroshot = "The answer is"
    args.direct_answer_trigger_for_zeroshot_cot = "The answer is"
    args.cot_trigger = "Let's think step by step."

    return args