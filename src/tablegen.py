from PIL import Image, ImageDraw, ImageFont
import random
import numpy as np
import time
from lorem.text import TextLorem
import lorem
from dice_roller import roll, fn_map, SAMPLES, roll_value
import string
import xml.etree.ElementTree as ET
import os
import math

class Tablegen():
    
    def __init__(self, width, height, compose_type, node):

        self.height = height
        self.width = width
        self.node = node
        self.compose_type = compose_type

        # --- general parameters which should be common for all configurations
        self.min_font = int(node.get('font_min_size', 14)) # 13
        self.base_font = max(self.min_font,self.height//80)
        self.min_tabletop = max(20,self.height//75)
        self.pad_font_min_size = int(roll_value(node.get('font_delta_size', 6)))
        # ---
        
        # --- can be loaded from a json file
        self.config = {}
        self.min_width = 50
        self.min_height = 50

        keys_file =  node.get('keys_path', None)
        assert keys_file
        with open(keys_file, 'r') as f:
            self.textheads = f.read()
            self.textheads = self.textheads.split('\n')

        if self.compose_type == 'plaintable':
            self.tabletitles = open(node.get('headers_path'),'r').read().split('\n')


        """
        keys_path (txt with keyval texts)
        headers_path (txt with top of the table only)

        table:
         - headers_path
         - keys_path
         - rows (dist)
         - cols (dist)
         - zero_width (bool, if True no borders are drawn)
         - text_p_isbold (prob uniform)
         - table_p_verticalsep (probability of using vertical separators)

        multicell:
         - keys_path
         - key_p_leftpos 
         - key_p_righpos
         - cell_p_iscolored (uiform if we have to color this)
         - cell_p_hasplaintext (no key-value, but only text)
         - cell_color (the specify color)
         - border_width (prob gauss)
         - text_p_isbold (prob uniform)
         - zero_width (bool, if True no borders are drawn)
         - cell_p_fullborders (put the border on all subcell template) ptobability
        """

        self.im = Image.new("L", (self.width, self.height), 255)

    def get_L_image(self):
        return self.im.convert("L")

    def text_wrap(self, text, font, max_width):
        lines = []
        if font.getsize(text)[0] <= max_width:
            lines.append(text) 
        else:
            words = text.split(' ')  
            i = 0
            while i < len(words):
                line = ''         
                while i < len(words) and font.getsize(line + words[i])[0] <= max_width:                
                    line = line + words[i] + " "
                    i += 1
                if not line:
                    line = words[i]
                    i += 1
                lines.append(line)    
        return lines

    def rect_draw(self, im, rect, width = 2, color = 0, border = 'full'):
        """
        Draw the border of a given rect with a specified style 
        """
        if self.node.get('zero_width') == False:
            rect = list(rect)
            rect[2] -= width//2
            rect[3] -= width//2
            draw = ImageDraw.Draw(im)
            if border == 'full':
                draw.line((rect[0],rect[1],rect[0]+rect[2],rect[1]), fill = color, width = width)
                draw.line((rect[0]+rect[2],rect[1],rect[0]+rect[2],rect[1]+rect[3]), fill = color, width = width)
                draw.line((rect[0],rect[1]+rect[3],rect[0]+rect[2],rect[1]+rect[3]), fill = color, width = width)
                draw.line((rect[0],rect[1],rect[0],rect[1]+rect[3]), fill = color, width = width)
            if border == 'vertical':
                draw.line((rect[0]+rect[2],rect[1],rect[0]+rect[2],rect[1]+rect[3]), fill = color, width = width)
                draw.line((rect[0],rect[1],rect[0],rect[1]+rect[3]), fill = color, width = width)
            if border == 'horizontal':
                draw.line((rect[0],rect[1],rect[0]+rect[2],rect[1]), fill = color, width = width)
                draw.line((rect[0],rect[1]+rect[3],rect[0]+rect[2],rect[1]+rect[3]), fill = color, width = width)
        return im
    
    def gen_celltext(self, font, rect):
        """
        Given a font (at specific size) and a rectagle to fill, generate the right amount of text
        """
        out = 'x'
        cnt = 0  
        go_len = True
        while go_len and cnt < 10000:      
            if roll() < 0.6:
                out = str(random.randint(23,9999))+','+ '{:0>2d}'.format(random.randint(0,99))
            else:
                lent = int(abs(np.random.normal(4,5)))
                out = ''.join(random.choice(string.ascii_letters) for x in range(lent))       
            cnt+=1
            go_len = font.getsize(out)[0] > rect[2]
        if cnt >= 10000:
            out = '00'

        return out

    def put_text(self, im, rect, mode = 'multiple', maxnum = 3):
        draw = ImageDraw.Draw(im)
        fontname = self.node.get('font')
            
        if roll() <= float(self.node.get('text_p_isbold')):
            fontname_keytop = fontname+' Bold'
            fontname = fontname + ' Bold'
        else:
            fontname_keytop = fontname
        
        font = ImageFont.truetype(fontname, self.base_font )
        font_keytop = ImageFont.truetype(fontname_keytop, self.base_font )
        line_height = font.getsize('hg')[1]
        line_height_keytop = font_keytop.getsize('hg')[1]
        font_large = ImageFont.truetype(fontname, int(self.base_font*1.8))#int(fsize*1.5) )
        line_height_large = font_large.getsize('hg')[1]
        font_tcell = ImageFont.truetype(fontname, self.min_font+self.pad_font_min_size)
        line_height_tcell = font_tcell.getsize('hg')[1]
        font_top = ImageFont.truetype(fontname, int(self.base_font*1.05))#int(rect[3]*0.6) )


        # --- drawing the multicell (when inside a big cell the loop stops and that cell is divided into smaller cells)
        if mode == 'multiple':
            if roll() <= self.node.get('cell_p_iscolored'):
                draw.rectangle((rect[0],rect[1],rect[2]+rect[0],rect[1]+rect[3]),fill=int(roll_value(self.node.get('cell_color'))))
                im = self.rect_draw(im, rect)
                draw = ImageDraw.Draw(im)

            aspect_ratio = rect[2]/rect[3]
            if aspect_ratio < 0.7:
                for z in range(0,maxnum):
                    num = self.gen_celltext(font,rect)
                    draw.text((rect[0]+5, rect[1]+z*rect[3]//maxnum),num,0,font=font)
            else:
                top_text = self.textheads[random.randint(0,len(self.textheads)-1)]
                if roll() <= self.node.get('uppercase_key', 0):
                    top_text = top_text.upper()
                num = self.gen_celltext(font_large,rect)

                toplen = font_keytop.getsize(top_text)[0]
                numlen = font_large.getsize(num)[0]
                
                p_mode = np.random.uniform(0,1)
                
                if roll() > 0.2 and rect[3] < 400:
                    # TOP BOTTOM MODE
                    # decide where to put the top and the bottom parts
                    p_top = roll()
                    if p_top < self.node.get('key_p_leftpos'):
                        xpostop = rect[0]+5
                    if p_top > 1-self.node.get('key_p_rightpos'):
                        xpostop = rect[0]+rect[2]-toplen-5
                    else:
                        xpostop = rect[0]+rect[2]//2 - toplen//2-10

                    p_num = roll()
                    if p_num < self.node.get('key_p_leftpos'):
                        xposnum = rect[0]+5
                    if p_num > 1-self.node.get('key_p_rightpos'):
                        xposnum = rect[0]+rect[2]-numlen-5
                    else:
                        xposnum = rect[0]+rect[2]//2 - numlen//2-10
                        
                    draw.text((xpostop, rect[1]+5),top_text,0,font=font_keytop)
                    draw.text((xposnum, rect[1]+rect[3]//2),num,0,font=font_large)
 
                else:
                    # RIGHT LEFT MODE 
                    num_rows = math.floor(rect[3]/(line_height_keytop*1.4))
                    for z in range(0,num_rows):
                        top_text = self.textheads[random.randint(0,len(self.textheads)-1)]
                        if roll() <= self.node.get('uppercase_key', 0):
                            top_text = top_text.upper()
                        num = self.gen_celltext(font_large,[rect[0],rect[1],rect[2]//3,rect[3]])

                        xposleft = rect[0]+int(rect[2]*0.01)
                        xposright = rect[0]+int(rect[2]*0.65)
                        draw.text((xposleft,rect[1]+5+z*line_height_keytop),top_text,0,font=font_keytop)
                        draw.text((xposright,rect[1]+5+z*line_height_keytop),num,0,font=font_keytop)

        # --- drawing the plaintable format (inner table only)
        if mode == 'tablecell':
            maxnum = max(1,rect[3]//line_height_tcell)
            maxiter = int(roll()*maxnum)
            h_shift = int(roll()*10)
            for z in range(0,maxiter):
                if roll() < 0.5:
                    num = str(random.randint(0,499))+','+ '{:0>2d}'.format(random.randint(0,99))
                else:
                    num = ''.join(random.choice(string.ascii_letters) for x in range(random.randint(2,6)))
                num = self.gen_celltext(font_tcell,rect)
                draw.text((rect[0]+h_shift,rect[1]+z*line_height_tcell),num,0,font=font_tcell)

        # --- drawing the header of the plaintable
        if mode == 'tabletop':
            title = self.tabletitles[random.randint(0,len(self.tabletitles)-1)]
            width = font_top.getsize(title)[0]
            starters = [rect[0]+3, rect[0]+(rect[2]//2)-(width//2), rect[0]+rect[2]-width-3]
            start = starters[random.randint(0,2)]
            start = starters[1]
            draw.text((start,rect[1]+2),title,0,font=font_top)

        # --- drawing the multicell cell (minimum element, no more subdivisions -- might be a large cell in which plaintext is put)
        if mode == 'cell':
            if roll() <= self.node.get('cell_p_iscolored'):
                draw.rectangle((rect[0],rect[1],rect[2]+rect[0],rect[1]+rect[3]),fill=int(roll_value(self.node.get('cell_color'))))
                im = self.rect_draw(im, rect)
                draw = ImageDraw.Draw(im)
                
            
            if roll() <= 1-self.node.get('cell_hasplaintext'): # very likely
                if rect[3]>rect[2]:
                    draw.text((rect[0]+5,rect[1]+5),self.textheads[random.randint(0,len(self.textheads)-1)],0,font=font)
                    num = self.gen_celltext(font_large, rect)
                    draw.text((rect[0]+5,rect[1]+rect[3]-line_height_large),num,0,font=font_large)
                else:
                    shift_up = random.randint(0,line_height_large//2)
                    idx = random.randint(0,len(self.textheads)-1)
                    draw.text((rect[0]+5,rect[1]+rect[3]//2-line_height_large//2+shift_up),self.textheads[idx],0,font=font_large)
                    num = self.gen_celltext(font_large, rect)
                    draw.text((rect[0]+10+font_large.getsize(self.textheads[idx])[0],rect[1]+rect[3]//2-line_height_large//2+shift_up),num,0,font=font_large)
            else:
                text = lorem.paragraph()
                textlines = self.text_wrap(text, font, rect[2])
                line_height = font.getsize('glA')[1]*1.2
                for z in range(0,len(textlines)):
                    if 5+(z+1)*line_height > rect[3]:
                        break
                    draw.text((rect[0]+5,rect[1]+5+z*line_height),textlines[z],0,font=font)
        return im

    # --- subcell template of the multicell mode
    def draw_subcell_template(self, im, rect):

        if roll() < self.node.get('cell_p_fullborders'):
            border = 'full'
        else:
            if roll() < 0.5:
                border = 'horizontal'
            else:
                border = 'vertical'

        width = int(roll_value(self.node.get('border_width')))
        maxnum = random.randint(2,4)

        im = self.rect_draw(im,rect)
        if rect[2] > rect[3]: # fat rectangle, split in columns
            ncells = random.randint(2,int(8)) # param to be fixed
            cell_size = rect[2]//ncells
            pad = 0

            for z in range(0,ncells):
                if z == ncells-1:
                    pad = rect[2]-cell_size*ncells

                im = self.rect_draw(im, (rect[0]+z*cell_size,rect[1],cell_size+pad,rect[3]),width = width, color = 0, border = border)
                im = self.put_text(im, (rect[0]+z*cell_size,rect[1],cell_size+pad,rect[3]), maxnum = maxnum)    

        else: # tall rectangle, split in rows
            ncells = random.randint(2,int(8)) # param to be fixed
            cell_size = rect[3]//ncells
            pad = 0
            for z in range(0,ncells):
                if z == ncells-1:
                    pad = rect[3]-cell_size*ncells
                im = self.rect_draw(im, (rect[0],rect[1]+z*cell_size,rect[2],cell_size+pad), width = width, color = 0, border = border)
                im = self.put_text(im, (rect[0],rect[1]+z*cell_size,rect[2],cell_size+pad), maxnum = maxnum)

        return im


    def make_multicell(self, im, rect, lev, max_lev):
        """
        Make a recursive multicell key-value like component
        - max_lev is the max recursion level (limit in order to have readable cells)
        """
        if rect[2] < self.min_width or rect[3] < self.min_height:
            return im

        if lev > max_lev or roll() < 0.1: # probability leaving big empty spaces (Fixed, small)
            im = self.rect_draw(im, rect)
            if roll() > 0.5:
                im = self.put_text(im,rect,mode = 'cell')
            return im

        if lev > max_lev-1 and roll()<0.8: # divide in subcells without further splits (Fixed)
            im = self.draw_subcell_template(im,rect)
            return im
        else:
            im = self.rect_draw(im, rect)

        if roll() < 0.5 and lev >= 2: # equiprobable vertical and horizontal splits
            # split vertically
            #xcoord = random.randint(0,rect[2])
            xcoord = abs(int(np.random.normal(rect[2]//2,20)))
            rect_left = (rect[0], rect[1], xcoord, rect[3]) 
            im = self.make_multicell(im,rect_left,lev+1, max_lev) # left

            rect_right = (rect[0]+xcoord, rect[1], rect[2]-xcoord, rect[3])
            im = self.make_multicell(im,rect_right,lev+1, max_lev) # right
        else:
            # split horizontally
            #ycoord = random.randint(0,rect[3])
            ycoord = abs(int(np.random.normal(rect[3]//2,20)))
            rect_up = (rect[0], rect[1], rect[2], ycoord)  
            im = self.make_multicell(im,rect_up,lev+1, max_lev) # left

            rect_down = (rect[0], rect[1]+ycoord, rect[2], rect[3]-ycoord)
            im = self.make_multicell(im,rect_down,lev+1, max_lev) # right

        return im

    def make_table(self, im, rect, n_rows, n_cols, top=True):
        """
        Make a plain table of n_rows x n_cols elements
        - if top = True, also make the table header
        """
        im = self.rect_draw(im, (rect[0],rect[1],rect[2],rect[3]),border='horizontal')
        if top:
            cell_size = rect[2]//n_cols
            pad = 0
            keep_text = abs(np.random.normal(0,1))
            for z in range(0,n_cols):
                height_top = max(self.min_tabletop,int(0.03*rect[3]))
                if z == n_cols-1:
                    pad = rect[2]-cell_size*n_cols
                im = self.rect_draw(im, (rect[0]+z*cell_size,rect[1],cell_size+pad,height_top))

                if keep_text > z/n_cols:
                    im = self.put_text(im, (rect[0]+z*cell_size,rect[1],cell_size+pad,height_top),mode='tabletop')  
            rect = (rect[0],rect[1]+height_top,rect[2],rect[3]-height_top)

        cell_height = rect[3]//n_rows
        cell_width = rect[2]//n_cols
        pad_height = 0

        keep_text = roll()
        rolled = roll()
        if rolled < self.node.get('table_p_verticalsep'):
            border = 'vertical'
        else:
            if rolled < 0.5:
                border = 'full'
            else:
                border = 'horizontal'

        for r in range(0,n_rows):
            if r == n_rows-1:
                    pad_height = rect[3]-cell_height*n_rows
            pad_width = 0

            set_text = False
            if keep_text > 0.7*r/(n_rows):
                    set_text = True  

            for c in range(0, n_cols):
                if c == n_cols-1:
                    pad_width = rect[2]-cell_width*n_cols

                #border = ('vertical' if roll() < self.node.get('table_p_verticalsep') else 'full')
                
                im = self.rect_draw(im, (rect[0]+c*cell_width,rect[1]+r*cell_height,cell_width+pad_width,cell_height+pad_height),border=border)
                if set_text:
                    im = self.put_text(im,(rect[0]+c*cell_width,rect[1]+r*cell_height+3,cell_width+pad_width,cell_height+pad_height),mode='tablecell')
        return im

    def calc_rc(self, rect):
        min_rows = 3
        min_cols = 2
        # at least 20px tall rows and 100px wide cols to improve readability (additional min-max check for the feasibility w.r.t. the page size)
        max_rows = int(rect[3]//50)
        max_cols = int(rect[2]//200)
        n_rows = max(min(max_rows,int(roll_value(self.node.get('rows')))),min_rows)
        n_cols = max(min(max_cols,int(roll_value(self.node.get('cols')))),min_cols)
        return n_rows, n_cols

    def show_image(self):
        assert(self.im is not None)
        self.im.show()

    def compose(self,im,rect):
        """
        Compose the specified type of content: plaintable,multicell 
        """
        if self.compose_type == 'plaintable':
            n_rows, n_cols = self.calc_rc(rect)
            self.im = self.make_table(im,(rect[0],rect[1],rect[2],rect[3]),n_rows,n_cols,roll() < self.node.get('table_p_header'))
        elif self.compose_type == 'multicell':
            self.im = self.make_multicell(im,(rect[0],rect[1],rect[2],rect[3]),0,2)



    # def compose(self, im,rect):
        
    #     if self.config['template'] == 'bustapaga':

    #         self.base_rect = rect
    #         im = self.rect_draw(im, rect)

    #         p_tableup = self.config['p_tableup']
    #         up_val = np.random.uniform(0.2,0.4)
    #         down_val = np.random.uniform(0.6,0.8)
    #         cut_up = int(np.random.normal(rect[3]*up_val,20))
    #         cut_down = int(np.random.normal(rect[3]*down_val,40))

    #         height_up = cut_up
    #         height_down = rect[3]-cut_down
    #         height_middle = rect[3]-height_down-height_up

    #         if random.random() < p_tableup:
    #             n_rows, n_cols = self.calc_rc(rect)
    #             im = self.make_table(im,(rect[0],rect[1],rect[2],height_up),n_rows,n_cols)
    #         else:
    #             im = self.make_multicell(im,(rect[0],rect[1],rect[2],height_up),0,2)

    #         n_rows, n_cols = self.calc_rc(rect)
    #         im = self.make_table(im,(rect[0],rect[1]+height_up,rect[2],height_middle),n_rows, n_cols)
    #         im = self.make_multicell(im,(rect[0],rect[1]+height_up+height_middle,rect[2],height_down),0,2)
    #         self.im = im

    #     if self.config['template'] == 'other':
    #         self.base_rect = rect
    #         im = self.rect_draw(im, rect)
    #         self.im = self.make_multicell(im,(rect[0],rect[1],rect[2],rect[3]),0,2)
            
    #     if self.config['template'] == 'simpletable':
    #         self.base_rect = rect
    #         im = self.rect_draw(im, rect)
    #         n_rows, n_cols = self.calc_rc(rect)
    #         self.im = self.make_table(im,(rect[0],rect[1],rect[2],rect[3]),n_rows,n_cols)

            # -- testing big box 
            #self.add2xml(rect)

            # if self.scale != 1:
            #     im = im.resize((self.s_width, self.s_height))
            #     bg = Image.new("L", (self.width,self.height), 255)
            #     w1 = self.pad_scale_x//2
            #     h1 = self.pad_scale_y//2
            #     bg.paste(im, (w1, h1))
            #     self.im = bg
            # else:
            #     self.im = im
            
            #self.im = spoil(im.convert("L"), options, crop = False)
