from __future__ import division, print_function, absolute_import
import time
import numpy as np
import os.path as osp
import datetime
import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from torchreid import metrics
from torchreid.utils import (AverageMeter, re_ranking, save_checkpoint,
                             visualize_ranked_results, tsne)
from torchreid.losses import DeepSupervision

import fitlog


class Engine(object):
    def __init__(self,
                 datamanager,
                 model,
                 optimizer=None,
                 scheduler=None,
                 use_gpu=True):
        self.datamanager = datamanager
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.use_gpu = (torch.cuda.is_available() and use_gpu)
        self.writer = None
        self.train_loader = self.datamanager.train_loader
        self.test_loader = self.datamanager.test_loader
        self.best_rank = 0

    def run(self,
            save_dir='log',
            max_epoch=0,
            start_epoch=0,
            print_freq=10,
            fixbase_epoch=0,
            open_layers=None,
            start_eval=0,
            eval_freq=-1,
            test_only=False,
            dist_metric='euclidean',
            normalize_feature=False,
            visrank=False,
            visrank_topk=10,
            use_metric_cuhk03=False,
            ranks=[1, 5, 10, 20],
            rerank=False):

        if visrank and not test_only:
            raise ValueError(
                'visrank can be set to True only if test_only=True')

        if test_only:
            self.test(0,
                      dist_metric=dist_metric,
                      normalize_feature=normalize_feature,
                      visrank=visrank,
                      visrank_topk=visrank_topk,
                      save_dir=save_dir,
                      use_metric_cuhk03=use_metric_cuhk03,
                      ranks=ranks,
                      rerank=rerank)
            return

        if self.writer is None:
            self.writer = SummaryWriter(log_dir=save_dir)

        time_start = time.time()
        print('=> Start training')

        for epoch in range(start_epoch, max_epoch):
            self.train(epoch,
                       max_epoch,
                       self.writer,
                       print_freq=print_freq,
                       fixbase_epoch=fixbase_epoch,
                       open_layers=open_layers)

            if (epoch + 1) >= start_eval \
               and eval_freq > 0 \
               and (epoch+1) % eval_freq == 0 \
               and (epoch + 1) != max_epoch:
                rank1 = self.test(epoch,
                                  dist_metric=dist_metric,
                                  normalize_feature=normalize_feature,
                                  visrank=visrank,
                                  visrank_topk=visrank_topk,
                                  save_dir=save_dir,
                                  use_metric_cuhk03=use_metric_cuhk03,
                                  ranks=ranks)
                if (epoch + 20) > max_epoch:
                    self._save_checkpoint(epoch, rank1, save_dir)

        if max_epoch > 0:
            print('=> Final test')
            rank1 = self.test(epoch,
                              dist_metric=dist_metric,
                              normalize_feature=normalize_feature,
                              visrank=visrank,
                              visrank_topk=visrank_topk,
                              save_dir=save_dir,
                              use_metric_cuhk03=use_metric_cuhk03,
                              ranks=ranks)
            self._save_checkpoint(epoch, rank1, save_dir)

        elapsed = round(time.time() - time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print('Elapsed {}'.format(elapsed))
        if self.writer is not None:
            self.writer.close()

    def train(self):
        raise NotImplementedError

    def test(self,
             epoch,
             dist_metric='euclidean',
             normalize_feature=False,
             visrank=False,
             visrank_topk=10,
             save_dir='',
             use_metric_cuhk03=False,
             ranks=[1, 5, 10, 20],
             rerank=False):
        targets = list(self.test_loader.keys())

        for name in targets:
            domain = 'source' if name in self.datamanager.sources else 'target'
            print('##### Evaluating {} ({}) #####'.format(name, domain))
            query_loader = self.test_loader[name]['query']
            gallery_loader = self.test_loader[name]['gallery']
            rank1 = self._evaluate(epoch,
                                   dataset_name=name,
                                   query_loader=query_loader,
                                   gallery_loader=gallery_loader,
                                   dist_metric=dist_metric,
                                   normalize_feature=normalize_feature,
                                   visrank=visrank,
                                   visrank_topk=visrank_topk,
                                   save_dir=save_dir,
                                   use_metric_cuhk03=use_metric_cuhk03,
                                   ranks=ranks,
                                   rerank=rerank)

        return rank1

    @torch.no_grad()
    def _evaluate(self,
                  epoch,
                  dataset_name='',
                  query_loader=None,
                  gallery_loader=None,
                  dist_metric='euclidean',
                  normalize_feature=False,
                  visrank=False,
                  visrank_topk=10,
                  save_dir='',
                  use_metric_cuhk03=False,
                  ranks=[1, 5, 10, 20],
                  rerank=False):
        batch_time = AverageMeter()

        def _feature_extraction(data_loader):
            f_, pids_, camids_, imgs_paths = [], [], [], []
            for batch_idx, data in enumerate(data_loader):
                imgs, pids, camids, imgs_path = self._parse_data_for_eval(data)
                if self.use_gpu:
                    imgs = imgs.cuda()
                end = time.time()
                features = self._extract_features(imgs)
                batch_time.update(time.time() - end)
                features = features.data.cpu()
                f_.append(features)
                pids_.extend(pids)
                camids_.extend(camids)
                imgs_paths.append(imgs_path)
            f_ = torch.cat(f_, 0)
            pids_ = np.asarray(pids_)
            camids_ = np.asarray(camids_)
            return f_, pids_, camids_, imgs_paths

        print('Extracting features from query set ...')
        qf, q_pids, q_camids, q_img_paths = _feature_extraction(query_loader)
        print('Done, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))

        print('Extracting features from gallery set ...')
        gf, g_pids, g_camids, g_img_paths = _feature_extraction(gallery_loader)
        print('Done, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))
        # time1 = time.time()
        # tsne(gf, g_pids)
        # print(f'time passed {time.time() - time1} ...')

        print('Speed: {:.4f} sec/batch'.format(batch_time.avg))

        if normalize_feature:
            print('Normalzing features with L2 norm ...')
            qf = F.normalize(qf, p=2, dim=1)
            gf = F.normalize(gf, p=2, dim=1)

        print(
            'Computing distance matrix with metric={} ...'.format(dist_metric))
        distmat = metrics.compute_distance_matrix(qf, gf, dist_metric)
        distmat = distmat.numpy()

        if rerank:
            print('Applying person re-ranking ...')
            distmat_qq = metrics.compute_distance_matrix(qf, qf, dist_metric)
            distmat_gg = metrics.compute_distance_matrix(gf, gf, dist_metric)
            distmat = re_ranking(distmat, distmat_qq, distmat_gg)

        print('Computing CMC and mAP ...')
        cmc, mAP, mINP = metrics.evaluate_rank(
            distmat,
            q_pids,
            g_pids,
            q_camids,
            g_camids,
            use_metric_cuhk03=use_metric_cuhk03)

        fitlog.add_metric(cmc[0] * 100, name='Rank-1', step=epoch + 1)
        fitlog.add_metric(mAP * 100, name='mAP', step=epoch + 1)
        fitlog.add_metric(mINP * 100, name='mINP', step=epoch + 1)

        if cmc[0] > self.best_rank:
            self.best_rank = cmc[0]
            best_step = epoch + 1
            fitlog.add_best_metric({
                "Test": {
                    "Epoch": str(best_step),
                    "Rank-1": '{:.1%}'.format(self.best_rank, ),
                    "mAP": '{:.1%}'.format(mAP, ),
                    "mINP": '{:.1%}'.format(mINP, ),
                }
            })

        print('** Results **')
        print('mAP: {:.1%}'.format(mAP))
        print('CMC curve')
        for r in ranks:
            print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))
        print('mINP: {:.1%}'.format(mINP))

        if visrank:
            visualize_ranked_results(
                distmat,
                self.datamanager.return_query_and_gallery_by_name(
                    dataset_name),
                self.datamanager.data_type,
                width=self.datamanager.width,
                height=self.datamanager.height,
                save_dir=osp.join(save_dir, 'visrank_' + dataset_name),
                topk=visrank_topk)

        return cmc[0]

    def _compute_loss(self, criterion, outputs, targets):
        if isinstance(outputs, (tuple, list)):
            loss = DeepSupervision(criterion, outputs, targets)
        else:
            loss = criterion(outputs, targets)
        return loss

    def _extract_features(self, input):
        self.model.eval()
        return self.model(input)

    def _parse_data_for_train(self, data):
        imgs = data[0]
        pids = data[1]
        return imgs, pids

    def _parse_data_for_eval(self, data):
        imgs = data[0]
        pids = data[1]
        camids = data[2]
        imgs_path = data[3]
        return imgs, pids, camids, imgs_path

    def _save_checkpoint(self, epoch, rank1, save_dir, is_best=False):
        save_checkpoint(
            {
                'state_dict': self.model.state_dict(),
                'epoch': epoch + 1,
                'rank1': rank1,
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
            },
            save_dir,
            is_best=is_best)
