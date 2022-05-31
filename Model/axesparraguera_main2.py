import os
import logging
from datetime import datetime
import time
import numpy as np
import pdb
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
from torch.utils.data import WeightedRandomSampler

from dataset import SoccerNetClips, SoccerNetClipsTrain, SoccerNetClipsTesting #,SoccerNetClipsOld
from model import Model
from train import trainer, test, testSpotting, testSpottingEnsemble
from loss import NLLLoss
from loss import NLLLoss_weights

# for reproducibility
torch.manual_seed(1)
np.random.seed(1)

def main(args):

    logging.info("Parameters:")
    for arg in vars(args):
        logging.info(arg.rjust(15) + " : " + str(getattr(args, arg)))

    # create dataset
    if not args.test_only:

        if args.version == 1:
            dataset_Train = SoccerNetClips(path=args.SoccerNet_path, features=args.features, split=args.split_train, version=args.version, framerate=args.framerate, chunk_size=args.chunk_size*args.framerate, augment = False)
            dataset_Valid = SoccerNetClips(path=args.SoccerNet_path, features=args.features, split=args.split_valid, version=args.version, framerate=args.framerate, chunk_size=args.chunk_size*args.framerate)
            dataset_Valid_metric  = SoccerNetClips(path=args.SoccerNet_path, features=args.features, split=args.split_valid, version=args.version, framerate=args.framerate, chunk_size=args.chunk_size*args.framerate)
    
        if args.version == 2:
            saved_loader = False
            if saved_loader == False:
                dataset_Train = SoccerNetClipsTrain(path_baidu = '/data-net/datasets/SoccerNetv2/Baidu_features', 
                             path_audio = '/data-local/data1-hdd/axesparraguera/vggish',
                             #path_audio = '/home-net/axesparraguera/data/VGGFeatures', 
                             path_labels = "/data-net/datasets/SoccerNetv2/ResNET_TF2", 
                             path_store = "/data-local/data3-ssd/axesparraguera",
                             features_baidu = 'baidu_soccer_embeddings.npy',
                             features_audio = 'featA.npy',
                             #features_audio = 'VGGish.npy', 
                             stride = 1, split=["train"], version=2, 
                            framerate=1, chunk_size=3, augment = False, store = False)
                
                #dataset_Train = SoccerNetClips(path=args.SoccerNet_path, features=args.features, split=args.split_train, version=args.version, framerate=args.framerate, chunk_size=args.chunk_size*args.framerate)
                dataset_Valid = SoccerNetClips(path=args.SoccerNet_path, features=args.features, split=args.split_valid, version=args.version, framerate=args.framerate, chunk_size=args.chunk_size*args.framerate)
                #dataset_Valid_metric  = SoccerNetClips(path=args.SoccerNet_path, features=args.features, split=args.split_valid, version=args.version, framerate=args.framerate, chunk_size=args.chunk_size*args.framerate)
    dataset_Test  = SoccerNetClipsTesting(path=args.SoccerNet_path, features=args.features, split=args.split_test, version=args.version, framerate=args.framerate, chunk_size=args.chunk_size*args.framerate)
            
        
            
    # create model
    model = Model(weights=args.load_weights, input_size=args.num_features,
                  num_classes=dataset_Test.num_classes, chunk_size=args.chunk_size*args.framerate,
                  framerate=args.framerate, pool=args.pool).cuda()
    logging.info(model)
    total_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    parameters_per_layer  = [p.numel() for p in model.parameters() if p.requires_grad]
    logging.info("Total number of parameters: " + str(total_params))

    # create dataloader
    if not args.test_only:
        sample_strategy = False
        if saved_loader == False:
            if sample_strategy == True:
                sampler = WeightedRandomSampler(torch.from_numpy(dataset_Train.weights).type('torch.DoubleTensor'), len(dataset_Train.weights))
                train_loader = torch.utils.data.DataLoader(dataset_Train,
                    batch_size=args.batch_size,
                    num_workers=args.max_num_worker, pin_memory=True, sampler = sampler)
                #torch.save(train_loader, 'train_loader_sampler.pth')
            else:
                train_loader = torch.utils.data.DataLoader(dataset_Train,
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=args.max_num_worker, pin_memory=True)
                #torch.save(train_loader, 'train_loader.pth')
                
            val_loader = torch.utils.data.DataLoader(dataset_Valid,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.max_num_worker, pin_memory=True)
            #torch.save(val_loader, 'val_loader.pth')

            val_metric_loader = torch.utils.data.DataLoader(dataset_Valid,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.max_num_worker, pin_memory=True)
            #torch.save(val_metric_loader, 'val_metric_loader.pth')
        else:
            if sample_strategy == True:
                train_loader = torch.load('train_loader_sampler.pth')
            else:
                train_loader = torch.load('train_loader.pth')

            val_loader = torch.load('val_loader.pth')
            val_metric_loader = torch.load('val_metric_loader.pth')
    
    #checkpoint = torch.load(os.path.join("models", args.model_name, "model_chunk3_full.pth.tar"))
    #model.load_state_dict(checkpoint['state_dict'])
    

    # training parameters
    if not args.test_only:
        #class_weights1 = torch.from_numpy(dataset_Train.class_weights1).type('torch.DoubleTensor').cuda()
        #class_weights2 = torch.from_numpy(dataset_Train.class_weights2).type('torch.DoubleTensor').cuda()
        criterion = NLLLoss_weights()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.LR, 
                                    betas=(0.9, 0.999), eps=1e-08, 
                                    weight_decay=1e-5, amsgrad=True)
        #optimizer = torch.optim.SGD(model.parameters(), lr=args.LR,
        #                            momentum=0.9)


        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=args.patience)
        
        # start training
        trainer(args.SoccerNet_path, train_loader, val_loader, val_metric_loader, 
                model, optimizer, criterion, patience=1000,
                model_name=args.model_name,
                max_epochs=11, evaluation_frequency=100)

    # For the best model only
    checkpoint = torch.load(os.path.join("models", args.model_name, "model_chunk3.pth.tar"))
    model.load_state_dict(checkpoint['state_dict'])

    # test on multiple splits [test/challenge]
    n_ensemble_train = 0
    for split in args.split_test:
    #for split in ['test']:
        
        ensemble = False
        
        if ensemble:
            ensemble_method = 'MLP'
            if (ensemble_method == 'MLP') & (n_ensemble_train == 0):
                results = testSpottingEnsemble(args.SoccerNet_path, args.model_name, 'valid', NMS_threshold=args.NMS_threshold, ensemble_method=ensemble_method)
                n_ensemble_train += 1
            results = testSpottingEnsemble(args.SoccerNet_path, args.model_name, split, NMS_threshold=args.NMS_threshold, ensemble_method=ensemble_method)
        
        else:
            dataset_Test  = SoccerNetClipsTesting(path=args.SoccerNet_path, features=args.features, split=[split], version=args.version, framerate=args.framerate, chunk_size=args.chunk_size*args.framerate)
            print('Test loader')
            test_loader = torch.utils.data.DataLoader(dataset_Test,
                batch_size=1, shuffle=False,
                num_workers=1, pin_memory=True)
        
        
            results = testSpotting(args.SoccerNet_path, test_loader, model=model, model_name=args.model_name, NMS_window=args.NMS_window, NMS_threshold=args.NMS_threshold)
        if results is None:
            continue

        a_mAP = results["a_mAP"]
        a_mAP_per_class = results["a_mAP_per_class"]
        a_mAP_visible = results["a_mAP_visible"]
        a_mAP_per_class_visible = results["a_mAP_per_class_visible"]
        a_mAP_unshown = results["a_mAP_unshown"]
        a_mAP_per_class_unshown = results["a_mAP_per_class_unshown"]

        logging.info("Best Performance at end of training ")
        logging.info("a_mAP visibility all: " +  str(a_mAP))
        logging.info("a_mAP visibility all per class: " +  str( a_mAP_per_class))
        logging.info("a_mAP visibility visible: " +  str( a_mAP_visible))
        logging.info("a_mAP visibility visible per class: " +  str( a_mAP_per_class_visible))
        logging.info("a_mAP visibility unshown: " +  str( a_mAP_unshown))
        logging.info("a_mAP visibility unshown per class: " +  str( a_mAP_per_class_unshown))

    return

if __name__ == '__main__':


    parser = ArgumentParser(description='context aware loss function', formatter_class=ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--SoccerNet_path',   required=False, type=str,   default="/path/to/SoccerNet/",     help='Path for SoccerNet' )
    parser.add_argument('--features',   required=False, type=str,   default="ResNET_TF2_PCA512.npy",     help='Video features' )
    parser.add_argument('--max_epochs',   required=False, type=int,   default=1000,     help='Maximum number of epochs' )
    parser.add_argument('--load_weights',   required=False, type=str,   default=None,     help='weights to load' )
    parser.add_argument('--model_name',   required=False, type=str,   default="Pooling",     help='named of the model to save' )
    parser.add_argument('--test_only',   required=False, action='store_true',  help='Perform testing only' )

    parser.add_argument('--split_train', nargs='+', default=["train"], help='list of split for training')
    parser.add_argument('--split_valid', nargs='+', default=["valid"], help='list of split for validation')
    parser.add_argument('--split_test', nargs='+', default=["test", "challenge"], help='list of split for testing')


    parser.add_argument('--version', required=False, type=int,   default=2,     help='Version of the dataset' )
    parser.add_argument('--num_features', required=False, type=int,   default=512,     help='Number of input features' )
    parser.add_argument('--evaluation_frequency', required=False, type=int,   default=10,     help='Number of chunks per epoch' )
    parser.add_argument('--framerate', required=False, type=int,   default=2,     help='Framerate of the input features' )
    parser.add_argument('--chunk_size', required=False, type=int,   default=60,     help='Size of the chunk (in seconds)' )
    parser.add_argument('--pool',       required=False, type=str,   default="MAX", help='How to pool' )
    parser.add_argument('--NMS_window',       required=False, type=int,   default=20, help='NMS window in second' )
    parser.add_argument('--NMS_threshold',       required=False, type=float,   default=0.5, help='NMS threshold for positive results' )

    parser.add_argument('--batch_size', required=False, type=int,   default=32,     help='Batch size' )
    parser.add_argument('--LR',       required=False, type=float,   default=1e-03, help='Learning Rate' )
    parser.add_argument('--LRe',       required=False, type=float,   default=1e-06, help='Learning Rate end' )
    parser.add_argument('--patience', required=False, type=int,   default=4,     help='Patience before reducing LR (ReduceLROnPlateau)' )

    parser.add_argument('--GPU',        required=False, type=int,   default=-1,     help='ID of the GPU to use' )
    parser.add_argument('--max_num_worker',   required=False, type=int,   default=4, help='number of worker to load data')

    # parser.add_argument('--logging_dir',       required=False, type=str,   default="log", help='Where to log' )
    parser.add_argument('--loglevel',   required=False, type=str,   default='INFO', help='logging level')

    args = parser.parse_args()

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)

    # os.makedirs(args.logging_dir, exist_ok=True)
    # log_path = os.path.join(args.logging_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S.log'))
    os.makedirs(os.path.join("models", args.model_name), exist_ok=True)
    log_path = os.path.join("models", args.model_name,
                            datetime.now().strftime('%Y-%m-%d_%H-%M-%S.log'))
    logging.basicConfig(
        level=numeric_level,
        format=
        "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ])

    if args.GPU >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)


    start=time.time()
    logging.info('Starting main function')
    main(args)
    logging.info(f'Total Execution Time is {time.time()-start} seconds')
