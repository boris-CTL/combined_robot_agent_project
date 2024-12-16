
config = {
    'device': 'cuda:0',                             # device to run on
    'save_interval': 10,                            # Save weights at every ith interval (None, int)
    'log_interval': 10,                             # Logging interval for debugging (None, int)
    'log_video_interval': 10,                       # Logging interval for videos (int)
    'record_file': 'records.pkl',                   # File to save global records dictionary
    'algorithm': 'PPO',                      # current training algorithm: 'supervised', 'RL', 'supervisedRL', 'debug', 'output_dataset_split'
    'mode': 'train',                                # 'train', 'eval'
    'do_supervised': True,                          # do supervised training in supervisedRL algorithm if set True
    'do_RL': True,                                  # do RL training in supervisedRL algorithm if set True
    # config for logging
    'logging': {
        'log_file': 'run.log',                      # log file name
        'fmt': '%(asctime)s: %(message)s',          # logging format
        'level': 'DEBUG',                           # logger level
        'wandb': False,                              # enable wandb logging
    },
    # config for data loader
    'data_loader': {
        'num_workers': 0,                           # Number of parallel CPU workers
        'pin_memory': False,                        # Copy tensors into CUDA pinned memory before returning them
#        'collate_fn': lambda x: x,                 # collate_fn to get minibatch as list instead of tensor
        'drop_last': True,
    },
    # Random seed for numpy, torch and cuda (None, int)
    'seed': 123,
    'optimizer': {
        'name': 'adam',
        'params': {
            'lr': 5e-4,
        },
        'scheduler': {
            'step_size': 10,                        # Period of learning rate decay
            'gamma': .95,                           # Multiplicative factor of learning rate decay
        }
    },

    # config to control training
    'train': {
        'data': {                                   # Dictionary to control dataset characteristics
            'to_tensor': True,
            'use_pickled': True
        },
        'batch_size': 256,
        'shuffle': True,
        'max_epoch': 100,
    },
    # config to control validation
    'valid': {
        'data': {                                   # Dictionary to control dataset characteristics
            'to_tensor': True,
            'use_pickled': True
        },
        'batch_size': 256,
        'shuffle': True,
        'debug_samples': [3, 37, 54],               # sample ids to generate plots for (None, int, list)
    },
    # config to control testing
    'test': {
        'data': {                                   # Dictionary to control dataset characteristics
            'to_tensor': True,
            'use_pickled': True
        },
        'batch_size': 256,
        'shuffle': True,
    },
    # config to control evaluation
    'eval': {
        'usage': 'test',                            # what dataset to use {train, valid, test}
    },
    'loss': {
        'latent_loss_coef': 1.0,                    # coefficient of latent loss (beta) in VAE during SL training
        'condition_loss_coef': 1.0,                 # coefficient of condition policy loss during SL training
    },
    'dsl': {
        'use_simplified_dsl': False,                # reducing valid tokens from 50 to 31
        'max_program_len': 12, #45,                 # maximum program length
        'grammar': 'handwritten',                   # grammar type: [None, 'handwritten']
    },
    # FIXME: This is only for backwards compatibility to old parser, should be removed soon
    'policy': 'TokenOutputPolicy',                  # output one token at a time (Ignore for intention space)
    'env_name': 'karel',
    'gamma': 0.99,                                  # discount factor for rewards (default: 0.99)
    'recurrent_policy': True,                       # If True, use RNN in policy network
    'num_lstm_cell_units': 64,                      # RNN latent space size
    'two_head': False,                              # do we want two headed policy? Not for LEAPS
    'mdp_type': 'ProgramEnv1',                      # ProgramEnv1: only allows syntactically valid program execution
    #'mdp_type': 'ProgramEnv_option',               # ProgramEnv_option: only allows syntactically valid program execution
    'env_task': 'harvester',                          # VAE: program,  meta-policy: cleanHouse, harvester, fourCorners, randomMaze, stairClimber, topOff
    'reward_diff': True,                            # If True, differnce between rewards of two consecutive states will be considered at each env step, otherwise current environment reward will be considered
    'prefix': 'default',                            # output directory prefix
    'max_program_len': 12, #45,                     # maximum program length  (repeated)
    'mapping_file': None,                           # mapping_karel2prl.txt if using simplified DSL (Ignore of intention space)
    'debug': False,                                 # use this to debug RL code (provides a lot of debug data in form of dict)
    'input_height': 8,                              # height of state image
    'input_width': 8,                               # width of state image
    'input_channel': 8,                             # channel of state image
    'border_size': 4,
    #'wall_prob': 0.1,                              # p(wall/one cell in karel gird)
    'wall_prob': 0.25,                              # p(wall/one cell in karel gird)
    'num_demo_per_program': 10,                     # 'number of seen demonstrations' (repeated)
    'gt_sample_demo_period': 1,                     # gt sample period for gt program behavior reconstruction
    'max_demo_length': 100,                         # maximum demonstration length (repeated)
    'min_demo_length': 2,                           # minimum demonstration length (repeated)
    'action_type': 'action',                       # Ignore for intention space
    'obv_type': 'state',                          # Ignore for intention space
    'reward_type': 'dense_subsequence_match',       # sparse, extra_sparse, dense_subsequence_match, dense_frame_set_match, dense_last_state_match
    'reward_validity': False,                       # reward for syntactically valid programs (Ignore for intention space)
    'fixed_input': True,                            # use fixed (predefined) input for program reconstruction task
    'max_episode_steps': 200,                         # maximum steps in one episode before environment reset
    'max_pred_demo_length': 500,                    # maximum steps for predicted program demo in behavior reconstruction task
    'AE': False,                                    # using plain AutoEncoder instead of VAE
    'experiment': 'intention_space',                # intention_space or EGPS
    'grammar':'handwritten',                        # grammar type: [None, 'handwritten']
    'use_trainable_tensor': False,                  # If True, use trainable tensor instead of meta-controller
    'cover_all_branches_in_demos': True,            # If True, make sure to cover all branches in randomly generated program in ExecEnv1
    'final_reward_scale': False,
    'make_error': False, 
    'policy': 'MlpPolicy',
    'n_eval_episodes': 10,
    'max_program_steps': 1,
    'llm_program_num': 10,
    'llm_temperature': 0.7,
    'gpt': 'gpt-4-1106-preview',
    'model_name': 'meta-llama/Llama-2-7b-chat-hf',
}
