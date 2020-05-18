class Config(object):
    imgDirPath = './data/Orient/DemoRightUpper05CTPA'
    case_list_train = ['/right/']

    # Weight path or "none"
    weightFile = "none"
    # Loss visualization
    # ON or OFF
    tensorboard = False
    tensorboard_logsDir = "backup"

    # model save path
    backupDir = "backup"
    logFile = "backup/log.txt"

    max_epochs = 6000*12
    save_interval = 100*12
    gpus = [0]
    # multithreading
    num_workers = 2
    batch_size = 1  # batch_size was always set to 1 in original keras implementation

    # Solver params
    # adam or sgd
    solver = "adam"
    steps = [10000]
    scales = [0.1]
    learning_rate = 3e-4
    momentum = 0.9
    decay = 5e-4
    betas = (0.9, 0.98)

    # Net params
    in_channels = 1
    patch_sz = (32, 32, 5)
    model_name = 'AV_CNN_GCN'
    # GCN setting
    num_nodes = 128 # num_nodes was named to batch size in original keras implementation
    Num_classes = 2
    Num_neighbors = 2
    dp = 0.5

