"""Render the Chinese Markdown as a three-page research brief."""
from pathlib import Path
import re
from xml.sax.saxutils import escape
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from pypdf import PdfReader
# ReportLab defaults cover Japanese punctuation but omit several Chinese marks.
# Extend the constants only inside this process to avoid leading punctuation.
import reportlab.lib.textsplit as textsplit
import reportlab.platypus.paragraph as paragraph_module
chinese_cannot_start = '，。：；！？）》」』】…'
textsplit.ALL_CANNOT_START += chinese_cannot_start
paragraph_module.ALL_CANNOT_START += chinese_cannot_start

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT.parents[1]
OUT=WORK/'output/pdf/rope_quest_short_report_zh.pdf'
fonts=Path('/System/Library/Fonts/Supplemental')
for name,index in [('Body',6),('BodyBold',1),('BodyItalic',6),('BodyBoldItalic',1),('Heading',6),('HeadingBold',0)]:
    pdfmetrics.registerFont(TTFont(name,str(fonts/'Songti.ttc'),subfontIndex=index))
pdfmetrics.registerFontFamily('Body',normal='Body',bold='BodyBold',italic='BodyItalic',boldItalic='BodyBoldItalic')
pdfmetrics.registerFontFamily('Heading',normal='Heading',bold='HeadingBold')
NAVY=colors.HexColor('#283f50');INK=colors.HexColor('#20252a');GRAY=colors.HexColor('#5f6770')
styles={
 'body':ParagraphStyle('body',fontName='Body',fontSize=10.5,leading=14.6,textColor=INK,spaceAfter=5.5),
 'h1':ParagraphStyle('h1',fontName='HeadingBold',fontSize=19,leading=23,textColor=NAVY,spaceAfter=7),
 'h2':ParagraphStyle('h2',fontName='HeadingBold',fontSize=12.5,leading=17,textColor=NAVY,spaceBefore=8,spaceAfter=7,keepWithNext=True),
 'h3':ParagraphStyle('h3',fontName='HeadingBold',fontSize=11.5,leading=15.5,textColor=NAVY,spaceBefore=7,spaceAfter=5,keepWithNext=True),
 'caption':ParagraphStyle('caption',fontName='BodyItalic',fontSize=8.7,leading=12.2,textColor=GRAY,spaceBefore=3,spaceAfter=8),
 'meta':ParagraphStyle('meta',fontName='Heading',fontSize=8.4,leading=11,textColor=GRAY,spaceAfter=10),
 'cell':ParagraphStyle('cell',fontName='Body',fontSize=9.5,leading=13.2,textColor=INK),
 'cellhead':ParagraphStyle('cellhead',fontName='HeadingBold',fontSize=9.3,leading=13,textColor=NAVY),
}
for style in styles.values():
    style.wordWrap='CJK'

def inline(s):
    s=escape(s)
    def link(m):
        label,url=m.groups()
        if not url.startswith(('https://','http://')):
            url=str((ROOT/url).resolve())
        return f'<a href="{url}" color="#365d7b">{label}</a>'
    s=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',link,s)
    s=re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',s)
    s=re.sub(r'`([^`]+)`',r'<font name="Courier" size="9">\1</font>',s)
    s=s.replace("(16/L)^0.25", "(16/L)<super>0.25</super>")
    return s

width=A4[0]-88
story=[];lines=(ROOT/'short_report_zh.md').read_text().splitlines();i=0
while i<len(lines):
    line=lines[i].strip()
    if not line:i+=1;continue
    if line=='<!-- pagebreak -->':story.append(PageBreak());i+=1;continue
    if line.startswith('!['):
        m=re.match(r'!\[.*?\]\((.*?)\)',line)
        img=Image(str(ROOT/m.group(1)))
        img.drawWidth=min(width,200*img.imageWidth/img.imageHeight)
        img.drawHeight=img.imageHeight*img.drawWidth/img.imageWidth
        img.hAlign='CENTER'
        story.extend([Spacer(1,2),img]);i+=1;continue
    if line.startswith('|'):
        rows=[]
        while i<len(lines) and lines[i].strip().startswith('|'):
            cells=[c.strip() for c in lines[i].strip().strip('|').split('|')]
            if not all(re.fullmatch('[-:]+',c) for c in cells):rows.append(cells)
            i+=1
        cols=[width*.33,width*.67] if len(rows[0])==2 else [width*.47,width*.22,width*.31]
        data=[[Paragraph(inline(c),styles['cellhead' if j==0 else 'cell']) for c in r] for j,r in enumerate(rows)]
        table=Table(data,colWidths=cols,hAlign='LEFT',repeatRows=1)
        table.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('LINEABOVE',(0,0),(-1,0),.6,NAVY),('LINEBELOW',(0,0),(-1,0),.4,NAVY),('LINEBELOW',(0,-1),(-1,-1),.5,NAVY),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f7f8f9')])]))
        story.extend([table,Spacer(1,8)]);continue
    if line.startswith('# '):style='h1';line=line[2:]
    elif line.startswith('## '):style='h2';line=line[3:]
    elif line.startswith('### '):style='h3';line=line[4:]
    elif line.startswith('*图 '):style='caption';line=line.strip('*')
    elif line.startswith('三页中文'):style='meta'
    else:style='body'
    story.append(Paragraph(inline(line),styles[style]));i+=1

def decorate(c,doc):
    c.saveState()
    if doc.page>1:
        c.setFont('Heading',8);c.setFillColor(GRAY);c.drawString(44,A4[1]-27,'RoPE 与 Quest 的长度偏差 | 中文简报')
    c.setStrokeColor(colors.HexColor('#c9ced1'));c.setLineWidth(.35);c.line(44,33,A4[0]-44,33)
    c.setFont('Heading',7.7);c.setFillColor(GRAY)
    c.drawString(44,21,'2026 年 9 月 15 日 | 独立研究报告，未写入论文')
    c.drawRightString(A4[0]-44,21,f'{doc.page} / 3')
    c.restoreState()

doc=SimpleDocTemplate(str(OUT),pagesize=A4,rightMargin=44,leftMargin=44,topMargin=40,bottomMargin=43,title='RoPE 与 Quest 的长度偏差：三页中文研究简报',author='Research report',pageCompression=1)
doc.build(story,onFirstPage=decorate,onLaterPages=decorate)
reader=PdfReader(OUT)
print('PDF',OUT,'PAGES',len(reader.pages))
for j,page in enumerate(reader.pages,1):print('Page',j,'words',len(page.extract_text().split()))
assert len(reader.pages)==3,'Condensed brief must contain exactly three pages'
