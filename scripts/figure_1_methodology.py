"""Render Figure 1 from frozen publication payloads without scientific recomputation."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD_DIR = ROOT / 'publication/methodology_payload'
OUT = Path(os.environ.get('STELLAR_FIGURE_OUTPUT_DIR', ROOT / 'publication/figures'))
OUT.mkdir(parents=True, exist_ok=True)

WIDTH_IN = 260 / 25.4
HEIGHT_IN = 170 / 25.4
DPI = 600
FONT = 'DejaVu Sans'
COL = {
    'ink': '#20252B', 'muted': '#66717B', 'grid': '#D9DEE2',
    'spectral': '#2B6F92', 'representation': '#16827A',
    'experiment': '#6A7178', 'detector': '#C26A2E',
    'evidence': '#8B4B78', 'inspect': '#343A40', 'paper': '#FFFFFF',
}
REL = {
    'HARMONIC': ('#0072B2', '-'), 'SUBHARMONIC': ('#0072B2', '--'),
    'RATIONAL': ('#009E73', ':'), 'SUM_COMBINATION': ('#D55E00', '--'),
    'DIFFERENCE_COMBINATION': ('#CC79A7', '-.'), 'SIDEBAND_PAIR': ('#6A7178', '-'),
}


def rows(name):
    with (PAYLOAD_DIR / name).open(newline='') as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def add_box(ax, xy, wh, title, lines=(), color='ink', fill='#FFFFFF', lw=0.8,
            title_size=7.4, text_size=6.5, linestyle='-', tag=None):
    x, y = xy; w, h = wh
    patch = Rectangle((x, y), w, h, transform=ax.transAxes, facecolor=fill,
                      edgecolor=COL[color], lw=lw, linestyle=linestyle, zorder=2)
    ax.add_patch(patch)
    if tag:
        ax.text(x + .012, y + h - .018, tag, transform=ax.transAxes,
                ha='left', va='top', fontsize=7, color=COL[color], fontweight='bold')
    title_y = y + h*(.68 if lines else .52)
    ax.text(x + w/2, title_y, title, transform=ax.transAxes,
            ha='center', va='center', fontsize=title_size, color=COL['ink'],
            fontweight='semibold', zorder=3)
    if lines:
        ax.text(x + w/2, y + h*.25, '\n'.join(lines), transform=ax.transAxes,
                ha='center', va='center', fontsize=text_size, color=COL['ink'],
                linespacing=1.05, zorder=3)
    return patch


def arrow(ax, start, end, color='muted', style='-|>', connection='arc3,rad=0'):
    ax.add_patch(FancyArrowPatch(start, end, transform=ax.transAxes,
                 arrowstyle=style, mutation_scale=7, lw=.75,
                 color=COL[color], connectionstyle=connection, zorder=1))


def region(ax, y0, y1, label, title, color):
    ax.add_patch(Rectangle((.012, y0), .976, y1-y0, transform=ax.transAxes,
                 facecolor=COL[color]+'08', edgecolor=COL[color], lw=.8, zorder=0))
    ax.text(.022, y1-.018, label, transform=ax.transAxes, fontsize=10,
            fontweight='bold', color=COL[color], ha='left', va='top')
    ax.text(.048, y1-.018, title, transform=ax.transAxes, fontsize=8.8,
            fontweight='semibold', color=COL['ink'], ha='left', va='top')


payload = json.loads((PAYLOAD_DIR/'methodology_figure_payload.json').read_text())
assert payload['selected_object']['canonical_object_id'] == 'object:epic:60018752'
assert payload['selection_explicitly_excludes_anomaly_rank'] is True
assert payload['scientific_recomputation_performed'] is False
assert payload['views']['PISD_ONLY'] == 42
assert [payload['views'][k] for k in ('GRAPH_CORE_ONLY','GRAPH_EXTENDED_ONLY','GRAPH_ALL')] == [30,36,66]
assert payload['views']['PRIMARY'] == 'PISD_PLUS_GRAPH_ALL'
assert payload['experiment']['objects'] == 33 and payload['experiment']['holdouts'] == 30
assert payload['experiment']['reference_per_split'] == 26 and payload['experiment']['evaluation_per_split'] == 7

light = rows('methodology_figure_lightcurve.csv')
period = rows('methodology_figure_periodogram.csv')
peaks = rows('methodology_figure_peaks.csv')
nodes = [x for x in rows('methodology_figure_graph_nodes.csv') if x['display_in_panel']=='YES']
edges = [x for x in rows('methodology_figure_graph_edges.csv') if x['display_in_panel']=='YES']
assert len(light)==433 and len(nodes)==12 and len(edges)==6

mpl.rcParams.update({
    'font.family': FONT, 'font.size': 7.0, 'axes.titlesize': 7.4,
    'axes.labelsize': 6.5, 'xtick.labelsize': 6.2, 'ytick.labelsize': 6.2,
    'axes.linewidth': .55, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
})
fig = plt.figure(figsize=(WIDTH_IN, HEIGHT_IN), facecolor='white')
canvas = fig.add_axes([0,0,1,1]); canvas.set_axis_off()

region(canvas,.690,.982,'A','Spectral construction and representation','spectral')
region(canvas,.370,.672,'B','Frozen multi-view anomaly experiment','experiment')
region(canvas,.035,.350,'C','Rank evidence and scientific inspection','evidence')

# A: real data plots
ax_lc=fig.add_axes([.045,.765,.195,.145]);
t=np.array([float(x['time_bjd_minus_2450000']) for x in light]); f=np.array([float(x['relative_flux']) for x in light])
ax_lc.plot(t,f,color=COL['spectral'],lw=.55); ax_lc.scatter(t,f,s=1.2,color=COL['spectral'],alpha=.55,rasterized=False)
ax_lc.set_title('Observed light curve',pad=2);ax_lc.set_xlabel('BJD − 2450000 [d]',labelpad=1);ax_lc.set_ylabel('Relative flux',labelpad=1);ax_lc.grid(True,color=COL['grid'],lw=.35)
ax_lc.text(.02,.96,'EPIC 60018752\nillustrative object',transform=ax_lc.transAxes,ha='left',va='top',fontsize=6.3,color=COL['muted'])

ax_ls=fig.add_axes([.275,.765,.195,.145]); fr=np.array([float(x['frequency_day_inverse']) for x in period]);pw=np.array([float(x['power']) for x in period])
ax_ls.plot(fr,pw,color=COL['spectral'],lw=.65);ax_ls.set_title('Lomb–Scargle spectrum',pad=2);ax_ls.set_xlabel('Frequency [d⁻¹]',labelpad=1);ax_ls.set_ylabel('Power',labelpad=1);ax_ls.grid(True,color=COL['grid'],lw=.35)
dom=next(x for x in peaks if x['is_dominant']=='YES');ref=next(x for x in peaks if x['is_operational_reference']=='YES')
ax_ls.scatter(float(dom['frequency_day_inverse']),float(dom['power']),marker='^',s=21,facecolor=COL['detector'],edgecolor=COL['ink'],lw=.4,zorder=4,label='Dominant LS peak')
ax_ls.scatter(float(ref['frequency_day_inverse']),float(ref['power']),marker='D',s=20,facecolor='white',edgecolor=COL['representation'],lw=1,zorder=4,label='Operational reference')
ax_ls.legend(loc='upper right',fontsize=6.0,frameon=False,handletextpad=.3,borderpad=.1)

ax_pk=fig.add_axes([.505,.765,.195,.145]);
ax_pk.plot(fr,pw,color=COL['grid'],lw=.45); pf=np.array([float(x['frequency_day_inverse']) for x in peaks]);pp=np.array([float(x['power']) for x in peaks])
ax_pk.vlines(pf,0,pp,color=COL['muted'],lw=.35,alpha=.65);ax_pk.scatter(pf,pp,s=3,color=COL['ink'])
ax_pk.scatter(float(dom['frequency_day_inverse']),float(dom['power']),marker='^',s=20,color=COL['detector'],zorder=4)
ax_pk.scatter(float(ref['frequency_day_inverse']),float(ref['power']),marker='D',s=19,facecolor='white',edgecolor=COL['representation'],lw=1,zorder=4)
ax_pk.set_title('Resolution-aware interpretation',pad=2);ax_pk.set_xlabel('Frequency [d⁻¹]',labelpad=1);ax_pk.set_ylabel('Power',labelpad=1);ax_pk.grid(True,color=COL['grid'],lw=.35)
ax_pk.text(.98,.93,'89 retained peaks\nδfᵣ = 1/T\nwindow annotations\ntyped relations',transform=ax_pk.transAxes,ha='right',va='top',fontsize=6.1,color=COL['muted'])

arrow(canvas,(.242,.835),(.272,.835),'spectral');arrow(canvas,(.472,.835),(.502,.835),'spectral')

# A: PISD and graph branch
add_box(canvas,(.725,.805),(.12,.105),'PISD',('42 features','5 + 13 + 10 + 7 + 7'),color='representation',fill='#F4FAF9',title_size=8,text_size=6.5)
canvas.text(.785,.792,'Strength 5 · Morphology 13\nHarmonic 10 · Complexity 7 · Stability 7',transform=canvas.transAxes,ha='center',va='top',fontsize=6.0,color=COL['ink'])

ax_g=fig.add_axes([.865,.765,.105,.135]);ax_g.set_axis_off();nmap={n['node_id']:n for n in nodes}
for e in edges:
 a=nmap[e['source_node_id']];b=nmap[e['target_node_id']];c,ls=REL[e['edge_type']]
 ax_g.plot([float(a['layout_x']),float(b['layout_x'])],[float(a['layout_y']),float(b['layout_y'])],color=c,ls=ls,lw=.75,alpha=.8,zorder=1)
for n in nodes:
 marker='D' if n['is_operational_reference']=='True' else ('^' if n['is_dominant']=='True' else 'o')
 fc='white' if marker=='D' else (COL['detector'] if marker=='^' else '#C9D0D5')
 ec=COL['representation'] if marker=='D' else COL['ink']; ax_g.scatter(float(n['layout_x']),float(n['layout_y']),s=17 if marker!='o' else 9,marker=marker,facecolor=fc,edgecolor=ec,lw=.65,zorder=3)
ax_g.set_title('Canonical-reference\nneighbourhood',fontsize=6.4,pad=1)
canvas.text(.89,.751,'full graph: 89 nodes / 17,521 edges',transform=canvas.transAxes,ha='center',va='top',fontsize=5.6,color=COL['muted'])
canvas.text(.89,.728,'Core 30 | Extended 36 | All 66',transform=canvas.transAxes,ha='center',va='top',fontsize=5.7,color=COL['ink'])
arrow(canvas,(.703,.835),(.721,.855),'representation');arrow(canvas,(.703,.825),(.86,.825),'representation',connection='arc3,rad=-.08')

# B: views, split, preprocessing, detectors
views=[('PISD','42'),('Core','30'),('Extended','36'),('Graph-All','66'),('PISD + Core','72'),('PISD + Extended','78'),('PISD + Graph-All','108')]
coords=[(.04,.525),(.145,.525),(.25,.525),(.355,.525),(.09,.445),(.215,.445),(.34,.445)]
for i,((name,dim),(x,y)) in enumerate(zip(views,coords)):
 primary=i==6
 width=.095 if i<4 else .115
 add_box(canvas,(x,y),(width,.065),name,(dim,'PRIMARY · predefined' if primary else ''),color='representation',fill='#EAF5F3' if primary else '#FFFFFF',lw=1.25 if primary else .65,title_size=6.1,text_size=5.8 if primary else 6.1)
canvas.text(.04,.61,'Seven frozen representation views',transform=canvas.transAxes,fontsize=7.2,fontweight='semibold',color=COL['ink'])

add_box(canvas,(.525,.525),(.125,.085),'Object level',('33 objects','14,338 samples','aggregated by object'),color='experiment',title_size=6.8,text_size=5.6)
add_box(canvas,(.68,.525),(.14,.085),'Repeated holdouts',('30 deterministic','26 reference | 7 evaluation'),color='experiment',title_size=6.6,text_size=6.0)
arrow(canvas,(.655,.567),(.675,.567),'experiment')

canvas.text(.525,.505,'Preprocessing · fit on reference objects only',transform=canvas.transAxes,fontsize=6.4,fontweight='semibold')
for i,(lab,ls) in enumerate([('RAW','solid'),('STANDARD','dashed'),('ROBUST','dotted')]):
 add_box(canvas,(.525+i*.105,.455),(.095,.038),lab,(),color='experiment',linestyle={'solid':'-','dashed':'--','dotted':':'}[ls],title_size=5.9)

det=[('Isolation Forest','IF'),('Local Outlier Factor','LOF'),('Autoencoder','AE\nD → H → Z → H → D')]
for i,(name,detail) in enumerate(det):
 add_box(canvas,(.525+i*.15,.385),(.135,.055),name,(detail,),color='detector',fill='#FFF8F2',title_size=5.8,text_size=5.7)
canvas.text(.845,.565,'views × preprocessing\n× detector configurations',transform=canvas.transAxes,ha='left',va='center',fontsize=6.2,color=COL['muted'])

# C: rank chain and evidence
add_box(canvas,(.04,.215),(.095,.07),'Model scores',(),color='evidence',title_size=6.5)
add_box(canvas,(.15,.215),(.105,.07),'Ordinal ranks',('1 = most unusual',),color='evidence',title_size=6.5,text_size=5.8)
add_box(canvas,(.27,.205),(.16,.09),'Normalized ranks',('R = 1 − (r − 1)/(N − 1)','1 = most unusual'),color='evidence',fill='#FBF4F9',title_size=6.5,text_size=6.1)
arrow(canvas,(.142,.235),(.157,.235),'evidence');arrow(canvas,(.262,.235),(.277,.235),'evidence')

evidence=[('Cross-model\nagreement',('Spearman ρ · Kendall τ','Kendall W')),('Consensus',('median','normalized rank')),('Robustness',('representation sensitivity','PRIMARY ↔ FULL','perturbations; displacement'))]
for i,(title,lines) in enumerate(evidence):
 add_box(canvas,(.455+i*.13,.19),(.117,.105),title,lines,color='evidence',title_size=6.0,text_size=5.5 if i==2 else 5.9,linestyle=['-','--','-.'][i])
arrow(canvas,(.432,.235),(.45,.235),'evidence')

add_box(canvas,(.855,.205),(.12,.085),'Frozen inspection\ncohort',('prioritized for','inspection'),color='inspect',title_size=5.8,text_size=5.8)
arrow(canvas,(.832,.247),(.85,.247),'evidence')

add_box(canvas,(.34,.07),(.245,.075),'Internal scientific inspection',('Light curve · Spectrum · PISD · Graph',),color='inspect',title_size=6.8,text_size=6.1)
canvas.plot([.915,.915,.50],[.203,.18,.18],transform=canvas.transAxes,color=COL['inspect'],lw=.75)
arrow(canvas,(.50,.18),(.50,.148),'inspect')
canvas.plot([.62,.62],[.055,.155],transform=canvas.transAxes,color=COL['inspect'],lw=1.2)
canvas.plot([.625,.625],[.055,.155],transform=canvas.transAxes,color=COL['inspect'],lw=.55)
canvas.text(.622,.158,'EVIDENCE FREEZE',transform=canvas.transAxes,ha='center',va='bottom',fontsize=6.5,fontweight='semibold',color=COL['inspect'])
add_box(canvas,(.68,.07),(.285,.075),'External catalogue / literature context',('annotates interpretation','does not alter anomaly ranking'),color='inspect',linestyle='--',title_size=6.5,text_size=6.1)
arrow(canvas,(.59,.107),(.615,.107),'inspect');arrow(canvas,(.63,.107),(.675,.107),'inspect')

# relation legend tucked under A graph
handles=[Line2D([0],[0],color=c,ls=ls,lw=1,label=k.replace('_',' ').title()) for k,(c,ls) in REL.items()]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.84,.687),ncol=3,frameon=False,fontsize=5.1,handlelength=1.35,columnspacing=.4,handletextpad=.2)

pdf=OUT/'figure_1_methodology.pdf';svg=OUT/'figure_1_methodology.svg';png=OUT/'figure_1_methodology.png'
fig.savefig(pdf,bbox_inches=None);fig.savefig(svg,bbox_inches=None);fig.savefig(png,dpi=DPI,bbox_inches=None)
plt.close(fig)

sources=['methodology_figure_payload.json','methodology_figure_lightcurve.csv','methodology_figure_periodogram.csv','methodology_figure_peaks.csv','methodology_figure_graph_nodes.csv','methodology_figure_graph_edges.csv']
manifest={
 'artifact_type':'publication_methodology_figure_render_manifest','manuscript_figure':'Figure 1',
 'selected_object':'EPIC 60018752','canonical_object_id':'object:epic:60018752',
 'selection_independent_of_anomaly_rank':True,'scientific_recomputation_performed':False,
 'source_payload_hashes':{x:sha(PAYLOAD_DIR/x) for x in sources},
 'figure_dimensions':{'width_mm':260,'height_mm':170,'width_in':WIDTH_IN,'height_in':HEIGHT_IN},
 'font':{'family':FONT,'minimum_effective_point_size':5.1,'main_text_target_pt':'7-9'},
 'layout_seed':None,'layout_seed_reason':'Persisted deterministic frequency/log-power coordinates; no stochastic layout',
 'graph_display_policy':payload['graph']['layout_policy']+'; persisted display_in_panel flags from Stage 22.14',
 'full_graph':{'nodes':89,'projected_edges':17521},'displayed_graph':{'nodes':len(nodes),'edges':len(edges)},
 'relation_types_displayed':sorted({x['edge_type'] for x in edges}),
 'contract':{'PISD':42,'Graph-Core':30,'Graph-Extended':36,'Graph-All':66,'views':7,'PRIMARY':'PISD_PLUS_GRAPH_ALL','objects':33,'holdouts':30,'reference':26,'evaluation':7,'detectors':['IF','LOF','AE'],'evidence_freeze_displayed':True},
 'outputs':{x.name:{'sha256':sha(x),'size_bytes':x.stat().st_size} for x in (pdf,svg,png)},
 'frozen_result_figures_modified':False,'public_release_modified':False,
}
(OUT/'figure_1_methodology_render_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'pdf':pdf.stat().st_size,'svg':svg.stat().st_size,'png':png.stat().st_size,'display_nodes':len(nodes),'display_edges':len(edges)},indent=2))
