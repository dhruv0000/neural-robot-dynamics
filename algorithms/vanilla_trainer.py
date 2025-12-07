import sys, os, time, shutil, yaml, numpy as np, torch, warp as wp
from typing import Optional
from torch.utils.data import DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')))

from envs.neural_environment import NeuralEnvironment
from models.models import ModelMixedInput
from models.jamba import JambaModel
from utils.datasets import BatchTransitionDataset, collate_fn_BatchTransitionDataset
from utils.evaluator import NeuralSimEvaluator
from utils.python_utils import set_random_seed, print_info, print_ok, print_white, print_warning, format_dict
from utils.torch_utils import num_params_torch_model, grad_norm
from utils.running_mean_std import RunningMeanStd
from utils.time_report import TimeReport, TimeProfiler
from utils.logger import Logger

class VanillaTrainer:
    def __init__(self, neural_env, cfg, model_checkpoint_path=None, device='cuda:0', novelty=None, wandb_project=None, wandb_name=None):
        self.novelty = novelty
        self.wandb_project = wandb_project
        self.wandb_name = wandb_name
        self.seed = cfg['algorithm'].get('seed', 0)
        self.device = device
        set_random_seed(self.seed)
        self.neural_env = neural_env
        self.neural_integrator = neural_env.integrator_neural

        # --- 1. Model Initialization ---
        if model_checkpoint_path is None:
            input_sample = self.neural_integrator.get_neural_model_inputs()
            
            # Helper to get input dimension from dict or tensor
            if isinstance(input_sample, dict):
                if 'states' in input_sample:
                    input_dim = input_sample['states'].shape[-1]
                else:
                    input_dim = list(input_sample.values())[0].shape[-1]
            else:
                input_dim = input_sample.shape[-1]

            if 'jamba' in cfg['network']:
                print(f"Initializing Jamba Model with Input Dim: {input_dim}")
                self.neural_model = JambaModel(input_dim, cfg['network'].get('d_model', 128), cfg['network'].get('n_layers', 4))
                self.neural_model.to(self.device)
            else:
                self.neural_model = ModelMixedInput(input_sample, self.neural_integrator.prediction_dim, cfg['inputs'], cfg['network'], device=self.device, novelty=self.novelty)
        else:
            checkpoint = torch.load(model_checkpoint_path, map_location=self.device)
            self.neural_model = checkpoint[0]
            self.neural_model.to(self.device)
        
        self.neural_integrator.set_neural_model(self.neural_model)

        # --- 2. Dataset Setup ---
        self.batch_size = int(cfg['algorithm']['batch_size'])
        self.dataset_max_capacity = cfg['algorithm']['dataset'].get('max_capacity', 100000000)
        self.num_data_workers = cfg['algorithm']['dataset'].get('num_data_workers', 4)
        
        # Initialize placeholders before calling get_datasets
        self.train_dataset = None
        self.valid_datasets = {} 
        self.collate_fn = None
        
        self.get_datasets(cfg['algorithm']['dataset'].get('train_dataset_path'), cfg['algorithm']['dataset'].get('valid_datasets'))
        
        # --- 3. Training Params ---
        # Default initialization
        self.lr_schedule = 'constant' 
        self.lr_start = 1e-3
        self.lr_end = 0.0

        if cfg.get('cli', {}).get('train', False):
            self.num_epochs = int(cfg['algorithm']['num_epochs'])
            self.num_iters_per_epoch = int(cfg['algorithm'].get('num_iters_per_epoch', -1))
            
            # Load Learning Rate Params
            self.lr_start = float(cfg['algorithm']['optimizer']['lr_start'])
            self.lr_end = float(cfg['algorithm']['optimizer'].get('lr_end', 0.))
            self.lr_schedule = cfg['algorithm']['optimizer']['lr_schedule']
            
            self.optimizer = torch.optim.Adam(self.neural_model.parameters(), lr=self.lr_start)
            
            # Logging
            self.log_dir = cfg['cli']['logdir']
            os.makedirs(self.log_dir, exist_ok=True)
            self.model_log_dir = os.path.join(self.log_dir, 'nn')
            os.makedirs(self.model_log_dir, exist_ok=True)
            self.logger = Logger()
            self.summary_log_dir = os.path.join(self.log_dir, 'summaries') 
            os.makedirs(self.summary_log_dir, exist_ok=True)
            self.logger.init_tensorboard(self.summary_log_dir)

            if self.wandb_project: 
                self.logger.init_wandb(self.wandb_project, self.wandb_name)
            
            self.save_interval = cfg['cli'].get("save_interval", 50)
            self.eval_interval = cfg['cli'].get("eval_interval", 1)
            self.log_interval = cfg['cli'].get("log_interval", 1) 
            
            # Gradient clipping params
            self.truncate_grad = cfg['algorithm'].get('truncate_grad', False)
            self.grad_norm = cfg['algorithm'].get('grad_norm', 1.0)
            
            # Compute Statistics
            if cfg['algorithm'].get("compute_dataset_statistics", True):
                print('Computing dataset statistics...')
                self.compute_dataset_statistics(self.train_dataset)
                if hasattr(self.neural_model, 'set_input_rms'):
                    self.neural_model.set_input_rms(self.dataset_rms)
                    self.neural_model.set_output_rms(self.dataset_rms['target'])

            # Create log files
            for valid_dataset_name in self.valid_datasets.keys():
                with open(os.path.join(self.model_log_dir, f'saved_best_valid_{valid_dataset_name}_model_epochs.txt'), 'w') as fp: fp.close()
            with open(os.path.join(self.model_log_dir, "saved_best_eval_model_epochs.txt"), 'w') as fp: fp.close()

        # --- 4. Evaluator Setup ---
        self.eval_mode = cfg['algorithm']['eval'].get('mode', 'sampler')
        self.num_eval_rollouts = cfg['algorithm']['eval'].get("num_rollouts", self.neural_env.num_envs)
        self.eval_render = cfg.get('cli', {}).get('render', False)
        self.eval_passive = cfg['algorithm']['eval'].get('passive', True)
        self.eval_horizon = cfg['algorithm']['eval'].get("rollout_horizon", 5)
        self.eval_dataset_path = cfg['algorithm']['eval'].get('dataset_path', None)

        self.evaluator = NeuralSimEvaluator(
            self.neural_env, 
            hdf5_dataset_path=self.eval_dataset_path if self.eval_mode == 'dataset' else None,
            eval_horizon=self.eval_horizon, 
            device=self.device
        )

    def get_datasets(self, train_path, valid_cfg):
        self.train_dataset = BatchTransitionDataset(self.batch_size, train_path, self.dataset_max_capacity, self.device)
        self.valid_datasets = {k: BatchTransitionDataset(self.batch_size, v, device=self.device) for k, v in valid_cfg.items()}
        self.batch_size = 1
        self.collate_fn = collate_fn_BatchTransitionDataset

    def compute_dataset_statistics(self, dataset):
        loader = DataLoader(dataset, batch_size=max(512, self.batch_size), collate_fn=self.collate_fn)
        self.dataset_rms = {}
        for data in loader:
            data = self.preprocess_data_batch(data)
            for k in data.keys():
                if k not in self.dataset_rms: self.dataset_rms[k] = RunningMeanStd(shape=data[k].shape[2:], device=self.device)
                self.dataset_rms[k].update(data[k], batch_dim=True, time_dim=True)

    def get_scheduled_learning_rate(self, iteration, total_iterations):
        if self.lr_schedule == 'constant': return self.lr_start
        elif self.lr_schedule == 'linear':
            ratio = iteration / total_iterations
            return self.lr_start * (1.0 - ratio) + self.lr_end * ratio
        elif self.lr_schedule == 'cosine':
            decay_ratio = iteration / total_iterations
            coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
            return self.lr_end + coeff * (self.lr_start - self.lr_end)
        return self.lr_start

    @torch.no_grad()
    def preprocess_data_batch(self, data):
        for k, v in data.items():
            if isinstance(v, dict): 
                for sk, sv in v.items(): data[k][sk] = sv.to(self.device)
            else: data[k] = v.to(self.device)
        data['contact_masks'] = self.neural_integrator.get_contact_masks(data['contact_depths'], data['contact_thicknesses'])
        self.neural_integrator.process_neural_model_inputs(data)
        data['target'] = self.neural_integrator.convert_next_states_to_prediction(data['states'], data['next_states'], self.neural_env.frame_dt)
        return data

    def compute_loss(self, data, train):
        pred = self.neural_model(data)
        
        # Determine weights
        if hasattr(self.neural_model, 'normalize_output') and self.neural_model.normalize_output:
            loss_weights = 1. / torch.sqrt(self.neural_model.output_rms.var + 1e-5)
        else:
            loss_weights = torch.ones(pred.shape[-1], device=pred.device)

        loss = torch.nn.MSELoss()(pred * loss_weights, data['target'] * loss_weights)
        return loss, {}

    def one_epoch(self, train, dataloader, dataloader_iter, num_batches, shuffle=False, info=None):
        if train: self.neural_model.train()
        else: self.neural_model.eval()
        sum_loss = 0
        with torch.set_grad_enabled(train):
            for _ in tqdm(range(num_batches)):
                try: data = next(dataloader_iter)
                except StopIteration:
                    if shuffle: self.train_dataset.shuffle()
                    dataloader_iter = iter(dataloader)
                    data = next(dataloader_iter)
                data = self.preprocess_data_batch(data)
                if train: self.optimizer.zero_grad()
                loss, _ = self.compute_loss(data, train)
                if train: 
                    loss.backward()
                    # Clip gradients
                    if self.truncate_grad:
                        clip_grad_norm_(self.neural_model.parameters(), self.grad_norm)
                    self.optimizer.step()
                sum_loss += loss
        return sum_loss / num_batches, {}, {}

    def train(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle=True, drop_last=True)
        train_iter = iter(train_loader)
        num_batches = len(train_loader) if self.num_iters_per_epoch == -1 else self.num_iters_per_epoch
        
        valid_loaders = {k: DataLoader(v, batch_size=self.batch_size, collate_fn=self.collate_fn) for k,v in self.valid_datasets.items()}
        valid_iters = {k: iter(v) for k,v in valid_loaders.items()}
        
        self.best_eval_error = np.inf
        
        self.time_report = TimeReport(cuda_synchronize = False)
        self.time_report.add_timers(['epoch', 'other', 'dataloader', 'compute_loss', 'backward', 'eval'])

        for epoch in range(self.num_epochs):
            self.logger.init_epoch(epoch)
            # Update LR
            self.lr = self.get_scheduled_learning_rate(epoch, self.num_epochs)
            for param_group in self.optimizer.param_groups: param_group['lr'] = self.lr

            if epoch > 0: self.one_epoch(True, train_loader, train_iter, num_batches, shuffle=True)
            for k, v in valid_loaders.items(): self.one_epoch(False, v, valid_iters[k], min(50, len(v)), info=k)
            
            if (epoch + 1) % self.eval_interval == 0: self.eval(epoch)
            
            # Flush logs
            self.logger.flush()
        
        # --- FIXED: Call finish() OUTSIDE the loop ---
        self.logger.finish() 

    @torch.no_grad()
    def eval(self, epoch):
        self.neural_model.eval()
        print('Evaluating...')
        error, _, stats = self.evaluator.evaluate_action_mode(
            num_traj=self.num_eval_rollouts, 
            eval_mode='rollout', 
            env_mode='neural', 
            trajectory_source=self.eval_mode, 
            render=self.eval_render, 
            passive=self.eval_passive
        )
        print(f"Eval Error: {stats['overall']['error(MSE)']}")
        
        if stats['overall']['error(MSE)'] < self.best_eval_error:
            self.best_eval_error = stats['overall']['error(MSE)']
            self.save_model('best_eval_model')

    def save_model(self, filename='best_model'):
        torch.save([self.neural_model, self.neural_env.robot_name], os.path.join(self.model_log_dir, f'{filename}.pt'))