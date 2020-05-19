import os
from utils import *
from config import Config
from utils.models import Av_CNN3D_model, Av_CNN_GCN_model#, Av_CNN_GCN_trans_model

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def inference(X_batch,Y_batch,NX_batch, config):
    global processed_batches

    model.eval()
    processed_batches = 0
    correct, total = 0, 0

    with torch.no_grad():

        processed_batches = processed_batches + 1
        X_batch, Y_batch, NX_batch = X_batch.cuda(), Y_batch.cuda(), NX_batch.cuda()

        output = model.forward(X_batch, NX_batch)
        # if len(output.shape) == 3:
        #     output = output.reshape(config.batch_size*config.num_nodes, -1)
        #     Y_batch = Y_batch.reshape(config.batch_size * config.num_nodes)
        pred = torch.argmax(output, dim=1)
        correct += torch.sum(pred.eq(Y_batch))
        total += output.shape[0]
        acc = np.array(correct.cpu())/total

        print('acc:', acc)

    print("done")

if __name__ == '__main__':
    config = Config()
    model_name = config.model_name
    use_cuda = torch.cuda.is_available()

    # path-----------------------------------------------------------------------------------
    if not os.path.exists(config.backupDir):
        os.mkdir(config.backupDir)

    # GPU-----------------------------------------------------------------------------------
    kwargs = {'num_workers': config.num_workers, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda:%s" % str(config.gpus[0]) if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(config.gpus[0])
        print("GPU is available!")
    else:
        print("GPU is not available!!!")

    # Load config params-----------------------------------------------------------------------
    if model_name == 'AV_CNN3D':
        usingNeighbors = False
        model = Av_CNN3D_model(droupout_rate=config.dp, number_class=config.Num_classes)
    elif model_name == 'AV_CNN_GCN':
        model = Av_CNN_GCN_model(cnnOFeat_len=10, gcnOFeat_len=config.Num_classes,
                                 gcnNumGaussian=6, gaussian_hidden_feat=3, number_neighbors=2, droupout_rate=0.5)

    model = model.cuda()
    # weights-----------------------------------------------------------------------------------
    if config.weightFile != 'none':
        model.load_weights(config.weightFile)
    else:
        model.seen = 0

    traingraphes = load_data(config.imgDirPath, config.case_list_train, Num_neighbor=config.Num_neighbors, shuffel=False)
    for data_idx in range(len(traingraphes)):
        graph = traingraphes[data_idx]
        patch_cnt = 0
        while patch_cnt + config.num_nodes <= len(graph.inds):
            X_batch, Y_batch, NX_batch = graph.next_node(node_num=config.num_nodes, WithNeighbor=True)
            Y_batch = np.argmax(Y_batch, axis=1)
            X_batch = torch.tensor(X_batch).float().permute(0, 3, 1, 2)
            Y_batch = torch.tensor(Y_batch).long()
            NX_batch = torch.tensor(NX_batch).float().permute(0, 1, 4, 2, 3).squeeze(dim=0)

            X_batch = X_batch[:, None, :, :]  # node count, channel, depth, width, height
            NX_batch = NX_batch[:, :, None, :, :]  # node count,neighbour,  channel, depth, width, height

            patch_cnt += config.num_nodes

            inference(X_batch,Y_batch,NX_batch, config)

    print('Done!')





