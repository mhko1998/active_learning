import os
import argparse
import random
import utils
import module
import neptune.new as neptune


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpus', default='1', type=str,
                        help='the number for used gpus')
    parser.add_argument('-m','--method',type=str, default='')
    parser.add_argument('-s','--save_path',type=str, default='')
    parser.add_argument('-t', '--teacher_path', type=str, default='')
    parser.add_argument('-p', '--startpoint', type=int, default=0)
    parser.add_argument('-d', '--data', type=str, default='cifar10')
    
    args = parser.parse_args()
    print(args)
    
    config = utils.read_conf('conf/'+args.data+'.json')

    max_epoch = int(config['epoch'])
    dataset_path = config['dataset']+args.data
    batch_size = int(config['batch_size'])
    save_path = config['save_path']+args.data+'/'+args.save_path
    teacher_path = config['save_path']+args.data+'/'+args.teacher_path
    teacher = args.teacher_path
    num_classes = int(config['num_classes'])
    ratio = float(config['ratio'])

    run = neptune.init(
    project="mhko1998/active",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjOGQ5Y2U4OC0xZWIzLTQyZjQtYWIyMy0wNTA5N2ExMzg2N2IifQ==",
) 
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    method = args.method
    startpoint = int(args.startpoint)

    
    os.environ['MASTER_ADDR'] = '172.27.183.200'
    a = random.randint(5000,9999)
    os.environ['MASTER_PORT'] = str(a)
    
    module.activerun(method, save_path, teacher_path, startpoint, max_epoch, dataset_path, batch_size, num_classes, ratio, run, teacher)
    
    run.stop()
    
        
if __name__=="__main__":
    main()