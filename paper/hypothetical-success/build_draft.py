"""Build matching Markdown, LaTeX, and a typeset review PDF from the same authoring source.

The PDF uses ReportLab, not a LaTeX compilation. Numerical figures are fictitious.
"""
from pathlib import Path
import json,re,html,os
os.environ.setdefault('MPLCONFIGDIR','/tmp/dexprompt-paper-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from reportlab.platypus import BaseDocTemplate,PageTemplate,Frame,Paragraph,Spacer,Image,Table,TableStyle,KeepTogether
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER,TA_JUSTIFY,TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors

ROOT=Path(__file__).resolve().parent
M=json.loads((ROOT/'manuscript.json').read_text()); refs={r['key']:i+1 for i,r in enumerate(M['references'])}
NUMERIC_NOTICE='HYPOTHETICAL SUCCESS DRAFT | ASSUMED RESULTS, NOT MEASURED'

def cite_text(s):
    return re.sub(r'\[@([^\]]+)\]',lambda m:'['+str(refs[m.group(1)])+']',s)

def tex_text(s):
    tokens=[]
    def reserve(m):
        tokens.append(m.group(1)); return 'CITATIONTOKEN'+str(len(tokens)-1)+'TOKENEND'
    s=re.sub(r'\[@([^\]]+)\]',reserve,s)
    replacements={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}','—':'---','–':'--'}
    s=''.join(replacements.get(c,c) for c in s)
    for i,k in enumerate(tokens):s=s.replace('CITATIONTOKEN'+str(i)+'TOKENEND',r'\cite{'+k+'}')
    return s

md=['# '+M['title'].replace('\n',' '),'', '> **'+M['notice']+'**','', 'This English manuscript deliberately uses completed-paper tense under the stated success assumption. Its methods and results are an authoring scenario, not the current project status.','',f"![Figure 1. Pipeline](figures/{M['hero']['name']}.png)", '', '**Figure 1.** '+M['hero']['caption'],'']
tex=[r'''% HYPOTHETICAL SUCCESS MANUSCRIPT. NOT AN EMPIRICAL REPORT.
% All assumed experimental methods/results are defined in manuscript.json and illustrative_results.json.
% Build with TEXINPUTS=..: pdflatex dexprompt_success.tex (twice).
\documentclass[letterpaper,10pt,conference]{ieeeconf}
\IEEEoverridecommandlockouts
\overrideIEEEmargins
\usepackage{graphicx}
\usepackage{amsmath,amssymb,booktabs,array,xcolor,fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[C]{\scriptsize\color{red!60!black}HYPOTHETICAL SUCCESS DRAFT --- ASSUMED RESULTS, NOT MEASURED}
\fancyfoot[C]{\thepage}
\setlength{\headheight}{13pt}
\renewcommand{\headrulewidth}{0pt}
\title{\LARGE\bf '''+tex_text(M['title']).replace(chr(10), chr(92)*2)+r'''}
\author{}
\begin{document}
\maketitle
\thispagestyle{fancy}
\begin{center}\footnotesize\color{red!60!black}'''+tex_text(M['notice'])+r'''\end{center}
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/01_pipeline.pdf}
\caption{'''+tex_text(M['hero']['caption'])+r'''}
\label{fig:pipeline}
\end{figure*}
''']
figure_number=1;table_number=0;equation_number=0;in_abstract=False
for b in M['blocks']:
    k=b['kind']
    if k=='heading':
        if in_abstract:tex.append(r'\end{abstract}');in_abstract=False
        title=b['text'];level=b['level']
        md += ['#'*(level+1 if level else 2)+' '+title,'']
        if level==0:tex.append(r'\begin{abstract}');in_abstract=True
        else:
            clean=re.sub(r'^[IVX]+\. |^[A-Z]\. ','',title)
            tex.append(('\\section{' if level==1 else '\\subsection{')+tex_text(clean)+'}')
    elif k=='paragraph':
        md += [cite_text(b['text']),'']; tex += [tex_text(b['text']),'']
    elif k=='equation':
        equation_number+=1
        md += ['$$',b['latex'], '$$', '']
        tex += [r'\begin{equation}',b['latex'],r'\label{eq:'+b['label']+'}',r'\end{equation}']
    elif k=='figure':
        figure_number+=1
        md += [f"![Figure {figure_number}](figures/{b['name']}.png)",'',f"**Figure {figure_number}.** "+b['caption'],'']
        tex += [r'\begin{figure}[t]',r'\centering',r'\includegraphics[width=\columnwidth]{figures/'+b['name']+'.pdf}',r'\caption{'+tex_text(b['caption'])+'}',r'\label{fig:'+b['name']+'}',r'\end{figure}']
    elif k=='table':
        table_number+=1
        md += [f"**Table {table_number}.** "+b['caption'],'','| '+' | '.join(b['headers'])+' |','|'+'|'.join([' --- ']*len(b['headers']))+'|']
        md += ['| '+' | '.join(row)+' |' for row in b['rows']];md += ['']
        cols='l'+'r'*(len(b['headers'])-1)
        tex += [r'\begin{table}[t]',r'\centering',r'\caption{'+tex_text(b['caption'])+'}',r'\label{tab:'+str(table_number)+'}',r'\resizebox{\columnwidth}{!}{%',r'\begin{tabular}{'+cols+'}',r'\toprule',' & '.join(tex_text(s) for s in b['headers'])+r' \\',r'\midrule']
        tex += [' & '.join(tex_text(s) for s in row)+r' \\' for row in b['rows']]
        tex += [r'\bottomrule',r'\end{tabular}}',r'\end{table}']
md+=['## References','']
tex += [r'\begin{thebibliography}{99}']
for i,r in enumerate(M['references'],1):
    md += [f"{i}. {r['text']}"]
    tex += [r'\bibitem{'+r['key']+'} '+tex_text(r['text'])]
tex += [r'\end{thebibliography}',r'\end{document}']
(ROOT/'manuscript.md').write_text('\n'.join(md)+'\n')
(ROOT/'dexprompt_success.tex').write_text('\n'.join(tex)+'\n')

# Typeset review PDF. All content and captions match the generated manuscript.
fontdir=Path('/usr/share/fonts/truetype/liberation2')
for name,suffix in [('PaperSerif','Regular'),('PaperBold','Bold'),('PaperItalic','Italic'),('PaperBoldItalic','BoldItalic')]:
    pdfmetrics.registerFont(TTFont(name,str(fontdir/f'LiberationSerif-{suffix}.ttf')))
pdfmetrics.registerFontFamily('PaperSerif',normal='PaperSerif',bold='PaperBold',italic='PaperItalic',boldItalic='PaperBoldItalic')
navy=colors.HexColor('#172b4d');red=colors.HexColor('#8d462b')
body=ParagraphStyle('Body',fontName='PaperSerif',fontSize=9.6,leading=11.3,alignment=TA_JUSTIFY,spaceAfter=4.4,splitLongWords=False)
abstract=ParagraphStyle('Abstract',parent=body,fontName='PaperBold',fontSize=9.4,leading=11.6)
heading=ParagraphStyle('Heading',parent=body,fontName='PaperBold',fontSize=10.1,leading=12,alignment=TA_CENTER,spaceBefore=9,spaceAfter=6,keepWithNext=True)
subheading=ParagraphStyle('Subheading',parent=body,fontName='PaperItalic',fontSize=9.8,leading=12,alignment=TA_LEFT,spaceBefore=6,spaceAfter=4,keepWithNext=True)
caption=ParagraphStyle('Caption',parent=body,fontSize=8.1,leading=9.7,alignment=TA_LEFT,spaceAfter=8)
cell=ParagraphStyle('Cell',parent=body,fontSize=8.0,leading=9.2,alignment=TA_LEFT,spaceAfter=0)
refstyle=ParagraphStyle('Ref',parent=body,fontSize=8.1,leading=9.8,spaceAfter=4,alignment=TA_LEFT)
titlestyle=ParagraphStyle('Title',parent=body,fontName='PaperBold',fontSize=20,leading=22.2,alignment=TA_CENTER)
noticestyle=ParagraphStyle('Notice',parent=caption,textColor=red,fontSize=8.0,leading=9.3,alignment=TA_LEFT)

W,H=612,792;left=45;right=45;gap=18;bottom=43;top=749
width=W-left-right;col=(width-gap)/2
hero_path=ROOT/'figures'/f"{M['hero']['name']}.png"
ih,iw=0,0
with PILImage.open(hero_path) as im: iw,ih=im.size
hero_height=width*ih/iw
# Establish first page geometry once, keeping the banner and full-width overview together.
title_p=Paragraph(html.escape(M['title']).replace('\n','<br/>'),titlestyle)
_,title_h=title_p.wrap(width,200)
notice_p=Paragraph(html.escape(M['notice']),noticestyle)
_,notice_h=notice_p.wrap(width,150)
hero_cap=Paragraph('<b>Fig. 1.</b> '+html.escape(M['hero']['caption']),caption)
_,hero_cap_h=hero_cap.wrap(width,150)
first_body_top=top-title_h-9-notice_h-9-hero_height-5-hero_cap_h-11

def decorate(canvas,doc,first=False):
    canvas.saveState()
    canvas.setFillColor(red);canvas.setFont('Helvetica-Bold',7.7)
    canvas.drawCentredString(W/2,H-22,NUMERIC_NOTICE)
    canvas.setStrokeColor(colors.HexColor('#d6dce5'));canvas.setLineWidth(.45)
    canvas.line(left,H-29,W-right,H-29)
    canvas.setFillColor(colors.HexColor('#77818f'));canvas.setFont('Helvetica',7.4)
    canvas.drawString(left,24,'DexPrompt | Hypothetical authoring edition | 12 September 2026')
    canvas.drawRightString(W-right,24,str(doc.page))
    if first:
        yy=top
        title_p.drawOn(canvas,left,yy-title_h);yy-=title_h+9
        notice_p.drawOn(canvas,left,yy-notice_h);yy-=notice_h+9
        canvas.drawImage(str(hero_path),left,yy-hero_height,width=width,height=hero_height,mask='auto');yy-=hero_height+5
        hero_cap.drawOn(canvas,left,yy-hero_cap_h)
    canvas.restoreState()

def frames(top_y,prefix):
    return [Frame(left,bottom,col,top_y-bottom,id=prefix+'L',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0),Frame(left+col+gap,bottom,col,top_y-bottom,id=prefix+'R',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)]

pdfpath=ROOT/'dexprompt_success.pdf'
doc=BaseDocTemplate(str(pdfpath),pagesize=(W,H),leftMargin=left,rightMargin=right,topMargin=43,bottomMargin=bottom,
 title=M['title'].replace('\n',' ')+' [HYPOTHETICAL SUCCESS DRAFT]',author='Anonymous hypothetical research draft',subject=M['notice'])
doc.addPageTemplates([PageTemplate(id='First',frames=frames(first_body_top,'F'),onPage=lambda c,d:decorate(c,d,True),autoNextPageTemplate='Normal'),PageTemplate(id='Normal',frames=frames(top,'N'),onPage=decorate)])

story=[];fig_no=1;tab_no=0;eq_no=0;is_abstract=False
for b in M['blocks']:
    k=b['kind']
    if k=='heading':
        is_abstract=b['level']==0
        story.append(Paragraph(html.escape(b['text']),heading if b['level']!=2 else subheading))
    elif k=='paragraph':story.append(Paragraph(html.escape(cite_text(b['text'])),abstract if is_abstract else body))
    elif k=='equation':
        eq_no+=1
        fig=plt.figure(figsize=(9,.6));fig.text(.01,.5,'$'+b['latex']+'$',ha='left',va='center',fontsize=17)
        eqpath=ROOT/'figures'/('equation_'+b['label']+'.png')
        fig.savefig(eqpath,dpi=220,bbox_inches='tight',pad_inches=.025,facecolor='white');plt.close(fig)
        with PILImage.open(eqpath) as im: ew,eh=im.size
        ewidth=min(col-23,ew/2.5)
        img=Image(str(eqpath),width=ewidth,height=ewidth*eh/ew)
        table=Table([[img,Paragraph('('+str(eq_no)+')',cell)]],colWidths=[col-22,22],hAlign='CENTER')
        table.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),0),('RIGHTPADDING',(0,0),(-1,-1),0)]))
        story += [table,Spacer(1,6)]
    elif k=='figure':
        fig_no+=1
        ip=ROOT/'figures'/(b['name']+'.png')
        with PILImage.open(ip) as im: iw,ih=im.size
        fw=col*b.get('width',1);fh=fw*ih/iw
        image=Image(str(ip),width=fw,height=fh)
        story.append(KeepTogether([Spacer(1,6),image,Spacer(1,4),Paragraph('<b>Fig. '+str(fig_no)+'.</b> '+html.escape(b['caption']),caption)]))
    elif k=='table':
        tab_no+=1
        cap=Paragraph('<b>TABLE '+['I','II','III','IV','V'][tab_no-1]+'.</b> '+html.escape(b['caption']),caption)
        n=len(b['headers'])
        if n==2: widths=[col*.65,col*.35]
        elif n==3:widths=[col*.49,col*.255,col*.255]
        else:widths=[col*.31,col*.23,col*.23,col*.23]
        vals=[[Paragraph('<b>'+html.escape(s)+'</b>',cell) for s in b['headers']]]+[[Paragraph(html.escape(s),cell) for s in row] for row in b['rows']]
        table=Table(vals,colWidths=widths,hAlign='CENTER')
        table.setStyle(TableStyle([('LINEABOVE',(0,0),(-1,0),.8,navy),('LINEBELOW',(0,0),(-1,0),.45,navy),('LINEBELOW',(0,-1),(-1,-1),.8,navy),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#f1f4f8')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),3),('RIGHTPADDING',(0,0),(-1,-1),3),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
        story.append(KeepTogether([Spacer(1,4),cap,table,Spacer(1,10)]))
story.append(Paragraph('References',heading))
for i,r in enumerate(M['references'],1):story.append(Paragraph('['+str(i)+'] '+html.escape(r['text']),refstyle))
doc.build(story)
print('Generated manuscript.md, dexprompt_success.tex, and dexprompt_success.pdf')
