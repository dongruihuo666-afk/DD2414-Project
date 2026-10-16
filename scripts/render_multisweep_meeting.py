#!/usr/bin/env python3
"""Build an evidence-linked, CPU-only meeting deck from existing experiment files."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BLUE, TEAL, ORANGE, PURPLE = '#2563eb', '#0d9488', '#d97706', '#7c3aed'
INK, GRAY, LIGHT = '#17263c', '#64748b', '#e2e8f0'
LABELS = ['Frozen backbone\n+ 1x1 probe', 'Trainable backbone\n+ 1x1 head',
          'End-to-end\n+ Simple-BEV decoder']
MODES = ['frozen-probe', 'unfrozen-probe', 'finetune']


class Evidence:
    def __init__(self):
        self.sources = {}

    def read(self, path, lines=False):
        path = Path(path)
        raw = path.read_bytes()
        key = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        self.sources[key] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        if not lines:
            return json.loads(raw)
        rows = []
        records = raw.splitlines()
        for i, line in enumerate(records):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if i != len(records)-1 or raw.endswith(b'\n'):
                    raise
                self.sources[key]['incomplete_final_line_ignored'] = True
        return rows

    def image(self, path):
        path = Path(path)
        raw = path.read_bytes()
        self.sources[str(path.relative_to(ROOT))] = dict(
            sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        return plt.imread(path)


def validation(result):
    if result.get('status') != 'complete':
        raise ValueError('Only complete runs may enter final performance figures')
    values = result['final_validation']
    if values['samples'] != 6019:
        raise ValueError('A final performance figure requires all 6,019 validation frames')
    tp, fp, fn = (values[k] for k in ['tp', 'fp', 'fn'])
    expected = dict(vehicle_iou=tp/(tp+fp+fn), precision=tp/(tp+fp),
                    recall=tp/(tp+fn), f1=2*tp/(2*tp+fp+fn))
    for key, number in expected.items():
        if not np.isclose(number, values[key], rtol=0, atol=1e-10):
            raise ValueError(f'Confusion counts disagree with {key}')
    return values


def block_mean(rows, key, block=100):
    points = [(r['epoch_progress'], r[key]) for r in rows if r.get(key) is not None]
    if not points:
        return np.array([]), np.array([])
    a = np.asarray(points)
    return tuple(np.asarray([a[i:i+block, j].mean() for i in range(0, len(a), block)])
                 for j in [0, 1])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study-dir', type=Path, default=ROOT/'fullsize_baseline/runs/multisweep_teacher_20261016')
    p.add_argument('--depth-dir', type=Path, default=ROOT/'fullsize_baseline/runs/depth_priority_20261016')
    p.add_argument('--reference-dir', type=Path, default=ROOT/'fullsize_baseline/runs/meeting_followup_20261016')
    p.add_argument('--output-dir', type=Path, default=ROOT/'artifacts/meeting_20261016')
    args = p.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    ev = Evidence()
    stamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    runs = {mode: args.study_dir/f'double_dino_5sweep_depth_none_{mode}_1epochs'
            for mode in ['pretrain', *MODES, 'object-motion']}
    results = {mode: ev.read(path/'summary.json') for mode, path in runs.items()}
    metrics = {mode: validation(results[mode]) for mode in MODES}
    official = {name: ev.read(args.study_dir/f'{name}/summary.json') for name in
                ['official_camera_112x192', 'official_camera_radar_5sweep_112x192']}
    official_metrics = [validation(result) for result in official.values()]
    references = {name: ev.read(args.reference_dir/name/'summary.json')
                  for name in ['official_camera', 'official_camera_radar_5sweep']}
    reference_metrics = [validation(result) for result in references.values()]
    manifests = {r['config']['manifest_sha256'] for r in
                 [*results.values(), *official.values(), *references.values()]}
    if len(manifests) != 1:
        raise ValueError('Compared experiments must share the dataset manifest')
    for mode in MODES:
        cfg = results[mode]['config']
        if cfg['nsweeps'] != 5 or cfg['depth_mode'] != 'none' or cfg['epochs'] != 1:
            raise ValueError('Unexpected downstream configuration')
    selected = ev.read(ROOT/'artifacts/fullsize_meeting_visuals.json')['samples']
    frame_rows = {mode: ev.read(path/'validation/epoch001/per_frame.jsonl', lines=True)
                  for mode, path in runs.items() if mode in MODES}
    for mode, rows in frame_rows.items():
        if len(rows) != 6019 or len({r['token'] for r in rows}) != 6019:
            raise ValueError('Missing or duplicate per-frame validation records')
        if [r['token'] for r in rows] != [r['token'] for r in frame_rows['finetune']]:
            raise ValueError('Paired comparisons require identical token ordering')
        for key in ['tp', 'fp', 'fn', 'tn']:
            if sum(row[key] for row in rows) != metrics[mode][key]:
                raise ValueError(f'{mode}: per-frame {key} differs from aggregate')
    frames = {mode: {r['token']: r for r in rows} for mode, rows in frame_rows.items()}
    pretrain_rows = ev.read(runs['pretrain']/'metrics.jsonl', lines=True)
    motion = results['object-motion']['final_validation']
    if results['object-motion']['status'] != 'complete' or motion['samples'] != 6019:
        raise ValueError('Object-motion evaluation is incomplete')
    depth = {}
    depth_downstream = {}
    for treatment in ['radar', 'radar-dav2']:
        path = args.depth_dir/f'double_dino_5sweep_depth_{treatment}_pretrain_1epochs'
        if (path/'status.json').exists():
            item = dict(status=ev.read(path/'status.json'), rows=[])
            if (path/'metrics.jsonl').exists():
                item['rows'] = ev.read(path/'metrics.jsonl', lines=True)
            if (path/'summary.json').exists():
                item['summary'] = ev.read(path/'summary.json')
            depth[treatment] = item
        depth_downstream[treatment] = {}
        for mode in ['frozen-probe','finetune']:
            path = args.depth_dir/f'double_dino_5sweep_depth_{treatment}_{mode}_1epochs/summary.json'
            if path.exists():
                result = ev.read(path)
                if result.get('status') == 'complete':
                    if result['config']['manifest_sha256'] not in manifests:
                        raise ValueError('Depth comparison manifest mismatch')
                    depth_downstream[treatment][mode] = validation(result)

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
        'axes.titlesize': 15, 'axes.labelsize': 12, 'axes.spines.top': False,
        'axes.spines.right': False, 'text.color': INK, 'axes.labelcolor': INK,
        'xtick.color': GRAY, 'ytick.color': GRAY, 'figure.facecolor': 'white'})
    figures, notes = [], []

    def slide(title, subtitle):
        fig = plt.figure(figsize=(16, 9))
        fig.text(.055, .935, title, fontsize=27, weight='bold', color=INK)
        fig.text(.055, .892, subtitle, fontsize=12, color=GRAY)
        return fig

    def finish(fig, name, take, speech):
        number = len(figures)+1
        fig.text(.055, .058, take, fontsize=13, color=INK,
                 bbox=dict(facecolor='#eff6ff', edgecolor='none', pad=10))
        fig.text(.055, .016, f'DD2414 | nuScenes trainval | Snapshot {stamp}', fontsize=9, color=GRAY)
        fig.text(.95, .016, f'{number:02d}', ha='right', fontsize=10, color=GRAY)
        figures.append((f'{number:02d}_{name}', fig))
        notes.append((number, fig.texts[0].get_text(), speech))

    fig = slide('What we built and what we can measure',
        'Full dataset: 28,130 training frames / 700 scenes; 6,019 validation frames / 150 scenes')
    ax = fig.add_axes([.045, .20, .91, .63]); ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch((x,y),w,h, boxstyle='round,pad=0.012',
                                  facecolor=color, edgecolor='none'))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=13,color=INK)
    def arrow(a, b, color=GRAY):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=16,color=color,lw=1.8))
    box(.02,.65,.25,.23,'Six cameras\nTrainable ResNet101\nRandom initialization', '#dbeafe')
    box(.02,.20,.25,.23,'Five radar sweeps\nBEVCar VoxelNet', '#ccfbf1')
    box(.36,.40,.25,.25,'Camera BEV + radar BEV\nFusion / shared BEV', '#ede9fe')
    box(.72,.67,.25,.20,'Frozen DINOv2 teacher\nImage + masked BEV losses', '#fef3c7')
    box(.72,.18,.25,.25,'Downstream GT evaluation\nFrozen 1x1 probe\nEnd-to-end segmentation', '#e0f2fe')
    arrow((.27,.75),(.36,.57)); arrow((.27,.32),(.36,.47))
    arrow((.61,.48),(.72,.32)); arrow((.72,.77),(.27,.77),ORANGE)
    arrow((.78,.67),(.60,.59),ORANGE)
    ax.text(.36,.16,'Current completed screen:\n112 x 192; seed 125\n1 pretraining epoch + 1 downstream epoch',fontsize=13,color=GRAY)
    finish(fig,'experiment','Two complementary questions: is the representation readable, and how well can the system adapt?',
        'We have implemented both DINO distillation terms and a five-sweep radar encoder. '
        'All completed student results here use a random camera initialization and 112 by 192 images. '
        'We pretrain once, then branch from that same checkpoint for the downstream experiments. '
        'The frozen probe measures linear accessibility of vehicle information; end-to-end training measures supervised adaptability.')

    fig = slide('Vehicle segmentation: two evaluation protocols',
        'Global valid-cell Vehicle IoU = TP / (TP + FP + FN); sigmoid threshold 0.5; all 6,019 validation frames')
    ax = fig.add_axes([.08,.28,.47,.51])
    values = [100*metrics[m]['vehicle_iou'] for m in MODES]
    bars = ax.bar(np.arange(3),values,color=[BLUE,PURPLE,TEAL],width=.62)
    ax.set(ylim=(0,38),ylabel='Vehicle IoU (%)',xticks=np.arange(3),xticklabels=LABELS)
    ax.set_title('Same student pretraining checkpoint',loc='left'); ax.grid(axis='y',alpha=.16)
    ax.bar_label(bars,fmt='%.2f%%',padding=8,fontsize=15,weight='bold')
    ax = fig.add_axes([.65,.28,.30,.51])
    refvals = [100*x['vehicle_iou'] for x in official_metrics]
    bars=ax.bar([0,1],refvals,color=['#94a3b8','#475569'],width=.58)
    ax.set(ylim=(0,38),xticks=[0,1],xticklabels=['Camera only','Camera + radar'],ylabel='Vehicle IoU (%)')
    ax.set_title('Official weights at 112 x 192',loc='left'); ax.grid(axis='y',alpha=.16)
    ax.bar_label(bars,fmt='%.2f%%',padding=8,fontsize=15)
    fig.text(.65,.16,'Official checkpoints were trained at higher resolution.\nThese are resolution-stress references, not matched retraining.',fontsize=11,color=GRAY)
    finish(fig,'segmentation','End-to-end adaptation improves IoU; this comparison does not isolate the benefit of DINO pretraining.',
        'The frozen probe reaches %.2f percent IoU. With the same linear head but a trainable backbone, '
        'IoU reaches %.2f percent; the full segmentation decoder reaches %.2f percent. '
        'The official weights fall to %.2f and %.2f percent when evaluated at our low resolution. '
        'We cannot use this table to claim superiority over matched, fully trained Simple-BEV.' %
        (values[0],values[1],values[2],refvals[0],refvals[1]))

    for i, sample in enumerate(selected):
        token=sample['sample_token']
        fig=slide(f'Prediction masks versus GT: fixed validation scene {i+1}',
            f'Token {token} | Identical scene, labels, threshold and BEV grid for both readouts')
        for j, mode in enumerate(['frozen-probe','finetune']):
            row=frames[mode][token]
            label='Frozen backbone + trained 1x1 probe' if mode=='frozen-probe' else 'Trainable backbone + Simple-BEV decoder'
            y=.53 if j==0 else .18
            fig.text(.065,y+.285,f'{label}  |  frame IoU {row["vehicle_iou"]*100:.2f}%',fontsize=14,weight='bold')
            ax=fig.add_axes([.055,y,.90,.28]); ax.axis('off')
            ax.imshow(ev.image(runs[mode]/f'validation/epoch001/mask_{token}.png'))
        finish(fig,f'masks_scene_{i+1}',
            'Error colors: green = correct vehicle; red = false positive; blue = missed vehicle. Black in mask = vehicle.',
            f'This is fixed scene {i+1}, selected before looking at these predictions. The top row uses '
            'the frozen representation and a trained linear probe; the bottom row allows full supervised adaptation. '
            'Read GT, probability, binary prediction, and errors from left to right. Blue regions expose missed vehicles; '
            'red regions show false detections. Each frame IoU is illustrative; headline IoU uses all validation cells.')

    fig=slide('What limits segmentation right now?',
        'Absolute confusion counts reveal errors that a single IoU number hides')
    ax=fig.add_axes([.08,.27,.43,.52]); x=np.arange(3); w=.34
    for offset,key,color,label in [(-w/2,'precision',BLUE,'Precision'),(w/2,'recall',ORANGE,'Recall')]:
        bars=ax.bar(x+offset,[100*metrics[m][key] for m in MODES],w,color=color,label=label)
        ax.bar_label(bars,fmt='%.1f',padding=4,fontsize=11)
    ax.set(ylim=(0,75),ylabel='Percent (%)',xticks=x,xticklabels=LABELS);ax.legend(frameon=False)
    ax.grid(axis='y',alpha=.16)
    ax=fig.add_axes([.64,.27,.30,.52])
    row=metrics['finetune']; total=row['tp']+row['fn']
    parts=[100*row['tp']/total,100*row['fn']/total]
    bottom=0
    for value,color,label in zip(parts,[TEAL,ORANGE],['Detected GT vehicle cells','Missed GT vehicle cells']):
        ax.bar([0],[value],bottom=bottom,color=color,width=.55,label=label)
        ax.text(0,bottom+value/2,f'{value:.1f}%',ha='center',va='center',color='white',fontsize=21,weight='bold')
        bottom+=value
    ax.set(ylim=(0,100),xticks=[0],xticklabels=['End-to-end model'],ylabel='Share of GT vehicle cells (%)')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.13),frameon=False,fontsize=10)
    finish(fig,'error_analysis','The end-to-end model misses %.1f%% of valid GT vehicle cells.'%parts[1],
        'Precision asks how often a predicted vehicle cell is correct; recall asks how many real vehicle cells we find. '
        'Recall is only %.2f percent for the end-to-end model, so %.2f percent of GT vehicle cells remain missed. '
        'High background accuracy would obscure these errors because vehicle cells are rare. '
        'The plots identify the error pattern, but do not prove whether initialization, resolution or training budget caused it.'%
        (100*row['recall'],parts[1]))

    fig=slide('Does the improvement hold across validation scenes?',
        'Per-scene IoU pools valid-cell counts within each scene; distributions cover the same 150 scenes')
    scenes={}
    for mode,rows in frame_rows.items():
        grouped={}
        for row in rows:
            counts=grouped.setdefault(row['scene_token'],dict(tp=0,fp=0,fn=0))
            for key in counts: counts[key]+=row[key]
        scenes[mode]={key:100*v['tp']/max(v['tp']+v['fp']+v['fn'],1) for key,v in grouped.items()}
    keys=sorted(scenes['finetune'])
    if len(keys)!=150 or any(sorted(scenes[m])!=keys for m in MODES):
        raise ValueError('Scene pairing requires the same complete 150 scenes')
    ax=fig.add_axes([.08,.27,.43,.52])
    bp=ax.boxplot([[scenes[m][s] for s in keys] for m in MODES],patch_artist=True,
                  showfliers=False,medianprops=dict(color=INK,linewidth=2))
    for box,color in zip(bp['boxes'],[BLUE,PURPLE,TEAL]): box.set(facecolor=color,alpha=.35)
    ax.set(xticks=[1,2,3],xticklabels=LABELS,ylabel='Per-scene Vehicle IoU (%)',ylim=(0,100));ax.grid(axis='y',alpha=.16)
    ax=fig.add_axes([.63,.27,.31,.52])
    delta=np.array([scenes['unfrozen-probe'][s]-scenes['frozen-probe'][s] for s in keys])
    ax.hist(delta,bins=20,color=PURPLE,alpha=.8);ax.axvline(0,color=INK,ls='--')
    ax.set(xlabel='Trainable minus frozen 1x1 IoU (pp)',ylabel='Scenes')
    wins=int((delta>0).sum())
    ax.set_title(f'Same 1x1 head: {wins}/150 scenes improve',loc='left',fontsize=14)
    finish(fig,'scene_distribution','Scene results describe one trained seed; they do not measure training-seed uncertainty.',
        f'The boxplots show scene-to-scene variation, rather than selecting a few favorable images. '
        f'Using the same linear head, allowing the backbone to update improves IoU in {wins} of 150 scenes. '
        'This demonstrates supervised adaptability. It does not isolate DINO from initialization and does not replace repeated training seeds.')

    fig=slide('Radar contributes to vehicle velocity estimation',
        'Frozen BEV features + trained linear velocity readout at GT box locations; all 6,019 validation frames')
    inputs=['matched','zero_velocity','empty','wrong_scene']; short=['Matched','Zero velocity','Empty','Wrong scene']
    colors=[TEAL,ORANGE,'#dc2626',PURPLE]
    for i,(group,label) in enumerate([('all','All vehicles'),('moving','Moving vehicles (> 1 m/s)'),('stationary','Stationary vehicles (<= 1 m/s)')]):
        ax=fig.add_axes([.065+i*.315,.29,.265,.49])
        vals=[motion['metrics'][m][group]['velocity_epe_mps'] for m in inputs]
        bars=ax.bar(np.arange(4),vals,color=colors,width=.65)
        ax.set(xticks=np.arange(4),xticklabels=short,ylabel='Velocity EPE (m/s), lower is better',ylim=(0,max(vals)*1.30))
        ax.tick_params(axis='x',rotation=25,labelsize=10); ax.bar_label(bars,fmt='%.2f',padding=5)
        ax.set_title(f'{label}\nn = {motion["metrics"]["matched"][group]["objects"]:,}',fontsize=13,loc='left')
        ax.grid(axis='y',alpha=.16)
        if group=='all':
            baseline=motion['baselines']['zero_prediction']['velocity_epe_mps']
            ax.axhline(baseline,color=GRAY,ls='--',label=f'Always zero: {baseline:.2f}')
            ax.legend(frameon=False,fontsize=10,loc='upper left')
    finish(fig,'radar_velocity','Aligned radar helps most for moving vehicles; stationary errors do not improve uniformly.',
        'Velocity EPE is the Euclidean error between predicted and GT XZ velocity vectors, in metres per second. '
        'For all vehicles the error is %.3f with matched radar, compared with %.3f for wrong-scene radar and %.3f with empty radar. '
        'The gain is larger for moving vehicles. The zero-velocity input slightly helps stationary vehicles, so the benefit is not uniform. '
        'GT vehicle boxes are supplied and a supervised readout is trained; this does not measure detection or nuScenes NDS.'%
        tuple(motion['metrics'][m]['all']['velocity_epe_mps'] for m in ['matched','wrong_scene','empty']))

    fig=slide('Pretraining learns the requested teacher objectives',
        'Five sweeps, no depth: full training epoch; curves use non-overlapping 100-update means')
    for i,(key,title,color) in enumerate([('image','Image-level DINO loss',BLUE),('bev','Radar-masked BEV DINO loss',TEAL),('radar_motion_auxiliary','Radar motion auxiliary',PURPLE)]):
        ax=fig.add_axes([.07+i*.315,.28,.265,.50])
        xx,yy=block_mean(pretrain_rows,key);ax.plot(xx,yy,color=color,lw=2.5)
        ax.set(xlim=(0,1),xlabel='Training epochs',ylabel='Training loss (lower is better)',title=title)
        ax.grid(alpha=.16)
        val=results['pretrain']['final_validation']['losses'][key]
        ax.text(.03,.08,f'Full validation: {val:.3f}',transform=ax.transAxes,color=color,fontsize=12)
    finish(fig,'pretraining','Teacher-feature matching and radar reconstruction improve; downstream GT evaluation remains necessary.',
        'Both requested DINO losses decrease during the full training epoch. The image term supervises the camera encoder '
        'and the BEV term supervises the fusion features under radar support. The motion reconstruction term is an auxiliary '
        'radar target. These losses are proxies, so their decline alone cannot demonstrate segmentation accuracy.')

    fig=slide('Depth experiment: current evidence and the next test',
        f'Live-run snapshot at {stamp}; displayed curves are training diagnostics at sparse radar anchors')
    axes=[fig.add_axes([.075,.46,.38,.31]),fig.add_axes([.565,.46,.38,.31])]
    description=[]
    for treatment,color,label in [('radar',TEAL,'Radar depth'),('radar-dav2',PURPLE,'Radar + Depth Anything V2')]:
        item=depth.get(treatment)
        if item and item['rows']:
            for ax,key in zip(axes,['radar_depth','depth_mae_m']):
                xx,yy=block_mean(item['rows'],key)
                ax.plot(xx,yy,color=color,lw=2.5,label=label)
            last=item['rows'][-1]
            description.append(f'{label}: {item["status"]["state"]}; {100*last["epoch_progress"]:.1f}% training exposure')
        else:
            description.append(f'{label}: no training records available at this snapshot')
    for ax,title,ylabel in zip(axes,['Sparse radar depth objective','Error at training radar anchors'],
                               ['Depth loss','Depth MAE (m)']):
        ax.set(title=title,xlabel='Training epochs',ylabel=ylabel,xlim=(0,1));ax.grid(alpha=.16)
        if ax.lines: ax.legend(frameon=False,fontsize=10)
        else: ax.text(.5,.5,'Results pending',ha='center',transform=ax.transAxes)
    fig.text(.075,.34,'\n'.join(description),fontsize=11,color=GRAY)
    table_ax=fig.add_axes([.075,.14,.87,.15]);table_ax.axis('off')
    table_rows=[['No depth',f'{100*metrics["frozen-probe"]["vehicle_iou"]:.2f}%',
                 f'{100*metrics["finetune"]["vehicle_iou"]:.2f}%']]
    for treatment,label in [('radar','Radar depth'),('radar-dav2','Radar + Depth Anything V2')]:
        table_rows.append([label,*[f'{100*depth_downstream[treatment][mode]["vehicle_iou"]:.2f}%'
            if mode in depth_downstream[treatment] else 'Pending'
            for mode in ['frozen-probe','finetune']]])
    table=table_ax.table(cellText=table_rows,
        colLabels=['Treatment','Frozen 1x1 Vehicle IoU','End-to-end Vehicle IoU'],
        cellLoc='center',bbox=[0,0,1,1]);table.auto_set_font_size(False);table.set_fontsize(11)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor(LIGHT)
        if r==0: cell.set_facecolor('#eff6ff');cell.set_text_props(weight='bold')
    finish(fig,'depth_snapshot','Depth training also changes image-to-BEV lifting; sparse-anchor errors are not independent dense-depth accuracy.',
        'This page is a timestamped snapshot of the ongoing depth experiment. The left curve is the sparse radar depth loss, '
        'and the right curve is training-anchor depth MAE. The same radar anchors provide supervision, so these errors are not '
        'independent dense-depth ground truth. Depth predictions actively weight lifting into BEV. '
        'We will compare no depth, radar depth, and radar plus Depth Anything V2 using the frozen probe and full downstream Vehicle IoU. '
        'Because lifting and supervision change together, we cannot attribute a future IoU gain to depth loss alone.')

    fig=slide('Next: a fair comparison at the standard resolution',
        'Current findings motivate controlled tests; they do not yet establish the root cause of the IoU gap')
    ax=fig.add_axes([.07,.48,.40,.31])
    labels=['Camera','Camera + 5-sweep radar'];x=np.arange(2);w=.34
    for offset,vals,color,label in [(-w/2,official_metrics,'#94a3b8','112 x 192 evaluation'),
                                  (w/2,reference_metrics,BLUE,'448 x 800 evaluation')]:
        bars=ax.bar(x+offset,[100*v['vehicle_iou'] for v in vals],w,color=color,label=label)
        ax.bar_label(bars,fmt='%.2f',padding=4,fontsize=11)
    ax.set(ylim=(0,68),xticks=x,xticklabels=labels,ylabel='Official-weight Vehicle IoU (%)')
    ax.legend(frameon=False,fontsize=10);ax.grid(axis='y',alpha=.16)
    fig.text(.55,.75,'Formal experiment',fontsize=21,weight='bold')
    plan=['448 x 800 cameras + ImageNet-pretrained ResNet101',
          'Trainable camera encoder + both DINO distillation terms',
          '10-sweep VoxelNet radar input; measured batch / accumulation',
          'Matched downstream budget, splits, augmentation and seeds',
          'Frozen 1x1 probe + end-to-end Vehicle IoU',
          'Same architecture without DINO as a pretraining control']
    for i,line in enumerate(plan):
        fig.text(.55,.69-i*.074,textwrap.fill(line,49),fontsize=12,va='top')
    fig.text(.075,.29,'Same official weights, different inference resolutions.\nThis tests resolution sensitivity; it does not isolate our training failure.',fontsize=12,color=GRAY)
    fig.text(.075,.19,'Corrected references from local summaries: %.2f%% camera / %.2f%% camera + radar.'%
             tuple(100*r['vehicle_iou'] for r in reference_metrics),fontsize=12,color=INK)
    finish(fig,'future_plan','Finish the depth screen, then test pretrained initialization and resolution under a matched protocol.',
        'Using the local official-checkpoint summaries, original-resolution Vehicle IoU is %.2f percent for camera-only and %.2f percent '
        'for camera plus radar. These are corrected directly from saved results. '
        'Reducing only inference resolution causes a large drop, establishing resolution sensitivity of these checkpoints. '
        'For our formal run we will use ImageNet-pretrained ResNet101, 448 by 800 inputs and ten radar sweeps. '
        'We also need a same-architecture control without DINO, matched supervised budgets and multiple seeds before crediting double distillation.'%
        tuple(100*r['vehicle_iou'] for r in reference_metrics))

    with PdfPages(out/'meeting_results.pdf') as pdf:
        for name,fig in figures:
            fig.savefig(out/f'{name}.png',dpi=160,facecolor='white')
            pdf.savefig(fig,facecolor='white')
    fig,axs=plt.subplots(4,3,figsize=(16,12));fig.subplots_adjust(wspace=.03,hspace=.06)
    for ax,(name,_) in zip(axs.flat,figures):
        ax.imshow(plt.imread(out/f'{name}.png'));ax.axis('off')
    fig.savefig(out/'contact_sheet.png',dpi=120,bbox_inches='tight',facecolor='#e2e8f0');plt.close(fig)
    for _,fig in figures: plt.close(fig)
    with (out/'metrics.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['configuration','resolution','vehicle_iou','precision','recall','f1','samples'])
        writer.writeheader()
        for label,res,row in [(mode,'112x192',metrics[mode]) for mode in MODES]+[
            (name,'112x192',m) for name,m in zip(official,official_metrics)]+[
            (name,'448x800',m) for name,m in zip(references,reference_metrics)]:
            writer.writerow(dict(configuration=label,resolution=res,**{key:row[key] for key in writer.fieldnames[2:]}))
    snapshot=dict(generated_utc=stamp,source_commit=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        metric='Global valid-cell TP / (TP + FP + FN); sigmoid threshold 0.5',
        validation_samples=6019,train_samples=28130,manifest_sha256=next(iter(manifests)),
        figures=[name+'.png' for name,_ in figures],sources=ev.sources,
        segmentation=metrics,official_low_resolution=official_metrics,
        official_original_resolution=reference_metrics,object_motion=motion,
        depth_status={k:v['status'] for k,v in depth.items()},
        depth_downstream=depth_downstream,
        selected_samples=selected,
        limitations=['One seed; one pretraining and one downstream epoch.',
            'Current camera encoder starts randomly; input resolution 112x192.',
            'Official weights are not retrained with matched budgets.',
            'Depth training anchors are not independent dense depth ground truth.',
            'GT object boxes are provided for velocity evaluation.',
            'Fixed qualitative sample selection is independent of model predictions.'])
    (out/'evidence.json').write_text(json.dumps(snapshot,indent=2)+'\n')
    speech=['# Meeting presentation notes','',f'Snapshot: {stamp}.','',
        'Open `meeting_results.pdf` in presentation mode. All individual PNGs are 16:9.',
        'Regenerate with `MPLCONFIGDIR=$PWD/.cache/matplotlib '
        '$HOME/miniconda3/envs/bev/bin/python scripts/render_multisweep_meeting.py`.',
        'This reads local results and uses CPU plotting only; it does not interrupt training.',
        'The output is a static snapshot; rerun after new experiment stages finish.','']
    for number,title,speech_text in notes:
        speech += [f'## {number:02d}. {title}','',speech_text,'']
    speech += ['## Provenance and checks','',
        '`evidence.json` records exact source paths and SHA-256 fingerprints. `metrics.csv` stores raw fractions.',
        'Generation verifies complete 6,019-frame results, confusion-count formulas, unique paired tokens, '
        'per-frame count totals, manifest identity, and the same 150 validation scenes.',
        'Original-resolution references are 47.247% / 55.495% in the current local summaries; '
        'earlier conversational values 47.56% / 55.81% were not used.','']
    (out/'SPEAKER_NOTES.md').write_text('\n'.join(speech))
    print(json.dumps(dict(output_dir=str(out),slides=len(figures),source_files=len(ev.sources),
                          completed_validation_samples=6019,paired_validation_scenes=len(keys)),indent=2))


if __name__=='__main__':
    main()
