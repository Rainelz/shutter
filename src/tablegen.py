from PIL import Image, ImageDraw, ImageFont
#import random
import numpy as np
import numpy.random as random
import time
#from lorem.text import TextLorem
import lorem
#import matplotlib.pyplot as plt
import string
import xml.etree.ElementTree as ET
import os
import math

class Tablegen():
    
    def __init__(self, width, height, fname='test', path='data', angle=0, scale=1):

        self.path = path
        self.xml_file = os.path.join(path,'Annotations/'+fname + '.xml')
        self.image_file = os.path.join(path,'JPEGImages/'+fname + '.png')

        self.height = height
        self.width = width
        self.fname = fname
        self.rot_angle = angle # angle used by the spoiler (needed to update the bbox coords)
        self.scale = scale
        self.s_height = round(self.scale*self.height)
        self.s_width = round(self.scale*self.width)
        self.pad_scale_x = self.width-self.s_width
        self.pad_scale_y = self.height-self.s_height

        # --- general parameters which should be common for all configurations
        self.base_font = self.height//80
        self.min_font = self.height//90
        self.min_tabletop = self.height//75
        self.pad_minfont = int(abs(random.normal(0,8)))
        # ---
        
        # --- can be loaded from a json file
        self.config = {}
        self.config['template'] = 'bustapaga'
        self.config['fontname'] = 'Courier New'
        self.config['textheads'] = ['Netto','Lordo','Sesso','Codice Fis.','Nome','Cognome','Residenza','Provincia','Stato','Euro','Lordo','Netto']
        self.config['tabletitles'] = ['CONTO','ESTRATTO','STIPENDIO','CONTRIBUTI','LORDO','TRATTENUTE','MESE','IRPEF']
        self.config['p_cellhastext'] = 0.8
        self.config['p_keycellbold'] = 0.5
        self.config['p_verticalsep_table'] = 0.5
        self.config['p_verticalsep_cells'] = 0.5
        self.config['p_left'] = 0.5 #  where to position inside subcells
        self.config['p_right'] = 0.1 #
        self.config['p_tableup'] = 0.5
        self.config['avg_rows'] = 3
        self.config['avg_cols'] = 5
        #self.config['']
        # ---
        
        self.xml_data = ET.Element('annotation')
        self.im = Image.new("L", (self.width, self.height), 255)
        self.xml_init()
        
    
        
    def xml_init(self):
        a = ET.SubElement(self.xml_data,'folder').text = 'Annotation'
        ET.SubElement(self.xml_data,'filename').text = self.fname+'.png'
        ET.SubElement(self.xml_data,'path').text = os.path.join(self.path,'Annotations/'+self.fname+'.png')
        sc = ET.SubElement(self.xml_data,'source')
        ET.SubElement(sc,'database').text = 'Custom table generator'
        sz = ET.SubElement(self.xml_data,'size')
        ET.SubElement(sz,'width').text = str(self.width)
        ET.SubElement(sz,'height').text = str(self.height)
        ET.SubElement(sz,'depth').text = '3'

    def outer_box(self, w, h, rect, alpha): 
        xc = w//2
        yc = h//2
        x = rect[0]
        y = rect[1]

        coords_x = [rect[0], rect[0]+rect[2], rect[0], rect[0]+rect[2]]
        coords_y = [rect[1], rect[1], rect[1]+rect[3], rect[1]+rect[3]]

        xr = []
        yr = []
        for z in range(len(coords_x)):
            xr.append(xc+(coords_x[z]-xc)*math.cos(alpha)+(coords_y[z]-yc)*math.sin(alpha))
            yr.append(yc-(coords_x[z]-xc)*math.sin(alpha)+(coords_y[z]-yc)*math.cos(alpha))

        return [min(xr), min(yr), max(xr), max(yr)]

        
    def add2xml(self, rect):
        if self.rot_angle != 0:
            alpha = math.pi * self.rot_angle / 180
            pd = 100 # base padding added by rotate (in spoil)
            rc_image = [pd, pd, pd+self.width, pd+self.height]
            rc = [rect[0]+pd, rect[1]+pd, rect[2], rect[3]]
            # compute the box which will be the cropped one
            busta_bbox = self.outer_box(self.im.size[0],self.im.size[1],rc_image, alpha)
            pad_x = busta_bbox[0]
            pad_y = busta_bbox[1]
            inner_bbox = self.outer_box(self.im.size[0],self.im.size[1],rc,alpha)
            bbox_val = [inner_bbox[0]-pad_x, inner_bbox[1]-pad_y, inner_bbox[2]-pad_x, inner_bbox[3]-pad_y] # togliere il pad aggiunto prima del crop
            # from within busta box compute the updated box with rect 
        # -----

        # Pascal VOC format
        obj = ET.SubElement(self.xml_data,'object')
        ET.SubElement(obj,'name').text = ('cell')
        ET.SubElement(obj,'pose').text = ('Unspecified')
        ET.SubElement(obj,'truncated').text = ('0')
        ET.SubElement(obj,'difficult').text = ('0')
        bbox = ET.SubElement(obj,'bndbox')

        shift_x = self.pad_scale_x//2
        shift_y = self.pad_scale_y//2

        
        if self.rot_angle == 0:
            ET.SubElement(bbox,'xmin').text = ("{:4.4f}".format(round(rect[0]*self.scale)+shift_x))
            ET.SubElement(bbox,'ymin').text = ("{:4.4f}".format(round(rect[1]*self.scale)+shift_y))
            ET.SubElement(bbox,'xmax').text = ("{:4.4f}".format(round(rect[0]+rect[2]*self.scale)+shift_x))
            ET.SubElement(bbox,'ymax').text = ("{:4.4f}".format(round(rect[1]+rect[3]*self.scale)+shift_y))
        else:
            ET.SubElement(bbox,'xmin').text = ("{:4.4f}".format(round(bbox_val[0]*self.scale)+shift_x))
            ET.SubElement(bbox,'ymin').text = ("{:4.4f}".format(round(bbox_val[1]*self.scale)+shift_y))
            ET.SubElement(bbox,'xmax').text = ("{:4.4f}".format(round(bbox_val[2]*self.scale)+shift_x))
            ET.SubElement(bbox,'ymax').text = ("{:4.4f}".format(round(bbox_val[3]*self.scale)+shift_y))
        
    def write_xml(self):
        # Pascal VOC format
        tree = ET.ElementTree(fname)
        tree.write(self.xml_file)
    
    def get_xml(self):
        return ET.ElementTree(self.xml_data)

    def get_L_image(self):
        return self.im.convert("L")
        
    def write_image(self):
        self.im.convert('RGB').save(self.image_file)

    def text_wrap(self, text, font, max_width):
        lines = []
        # If the width of the text is smaller than image width
        # we don't need to split it, just add it to the lines array
        # and return
        if font.getsize(text)[0] <= max_width:
            lines.append(text) 
        else:
            # split the line by spaces to get words
            words = text.split(' ')  
            i = 0
            # append every word to a line while its width is shorter than image width
            while i < len(words):
                line = ''         
                while i < len(words) and font.getsize(line + words[i])[0] <= max_width:                
                    line = line + words[i] + " "
                    i += 1
                if not line:
                    line = words[i]
                    i += 1
                # when the line gets longer than the max width do not append the word, 
                # add the line to the lines array
                lines.append(line)    
        return lines


    def rect_draw(self, im, rect, width = 1, color = 0, border = 'full'):
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
        out = 'xxxxxxxxxxxxxxxxxxxxxxxxx'
        cnt = 0  

        go_len = True
        while go_len and cnt < 10000:      
            if random.randint(0,100) < 70:
                out = str(random.randint(23,499))+','+ '{:0>2d}'.format(random.randint(0,99))
            else:
                lent = int(abs(random.normal(3,5)))
                chars = random.choice(list(string.ascii_letters), lent)
                out = ''.join(chars)
            cnt+=1
            go_len = font.getsize(out)[0] > rect[2]
        
        if cnt >= 10000:
            out = 'M'

        return out

    def put_text(self, im, rect, mode = 'multiple', maxnum = 3):
        draw = ImageDraw.Draw(im)
        self.add2xml(rect) # add this cell as annotation in the xml file
        
        fontname = self.config['fontname']
            
        if np.random.uniform(0,1) <= self.config['p_keycellbold']:
            fontname_keytop = fontname+' Bold'
        else:
            fontname_keytop = fontname
        

        font = ImageFont.truetype(fontname, self.base_font )
        font_keytop = ImageFont.truetype(fontname_keytop, self.base_font )
        line_height = font.getsize('hg')[1]
        font_large = ImageFont.truetype(fontname, int(self.base_font*1.15))#int(fsize*1.5) )
        line_height_large = font_large.getsize('hg')[1]
        #font_tcell = ImageFont.truetype(fontname, rect[3]//10)
        font_tcell = ImageFont.truetype(fontname, self.min_font+self.pad_minfont)
        line_height_tcell = font_tcell.getsize('hg')[1]
        font_top = ImageFont.truetype(fontname, int(self.base_font*1.05))#int(rect[3]*0.6) )
        line_height_top = font_top.getsize('hg')[1]
        
        table_titles = self.config['tabletitles']
        text_heads = self.config['textheads']

        if mode == 'multiple':
            if rect[3] == 0:
                return
            aspect_ratio = rect[2]/rect[3]
            if aspect_ratio < 0.7:

                for z in range(0,maxnum):
                    #num = str(random.randint(23,499))+','+ '{:0>2d}'.format(random.randint(0,99))
                    num = self.gen_celltext(font,rect)
                    draw.text((rect[0]+5, rect[1]+z*rect[3]//maxnum),num,0,font=font)
            else:
                top_text = text_heads[random.randint(0,len(text_heads)-1)]
                num = self.gen_celltext(font_large,rect)
                
                toplen = font_keytop.getsize(top_text)[0]
                numlen = font_large.getsize(num)[0]
                
                # decide where to put the top and the bottom parts
                p_top = random.uniform(0,1)
                if p_top < self.config['p_left']:
                    xpostop = rect[0]+5
                if p_top > 1-self.config['p_right']:
                    xpostop = rect[0]+rect[2]-toplen-5
                else:
                    xpostop = rect[0]+rect[2]//2 - toplen//2-10
                p_num = random.uniform(0,1)
                if p_num < self.config['p_left']:
                    xposnum = rect[0]+5
                if p_num > 1-self.config['p_right']:
                    xposnum = rect[0]+rect[2]-numlen-5
                else:
                    xposnum = rect[0]+rect[2]//2 - numlen//2-10
                    
                
                draw.text((xpostop, rect[1]+5),top_text,0,font=font_keytop)
                draw.text((xposnum, rect[1]+rect[3]//2),num,0,font=font_large)

        if mode == 'tablecell':
            vspace = (1+random.random())*line_height_tcell
            vspace = random.uniform(1,2)*line_height_tcell*4
            maxnum = max(1,rect[3]//2//line_height_tcell)

            for z in range(0,maxnum):

                if np.random.uniform() < 0.5:
                    num = str(random.randint(0,499))+','+ '{:0>2d}'.format(random.randint(0,99))
                else:
                    num = ''.join(random.choice(list(string.ascii_letters),random.randint(2,6)))
                
                num = self.gen_celltext(font_tcell,rect)
                #num = str(random.randint(23,499))+','+ '{:0>2d}'.format(random.randint(0,99))
                draw.text((rect[0]+5,rect[1]+z*line_height_tcell),num,0,font=font_tcell)

        if mode == 'tabletop':
            title = table_titles[random.randint(0,len(table_titles)-1)]
            width = font_top.getsize(title)[0]
            starters = [rect[0]+3, rect[0]+(rect[2]//2)-(width//2), rect[0]+rect[2]-width-3]
            start = starters[random.randint(0,2)]
            start = starters[1]
            draw.text((start,rect[1]+2),title,0,font=font_top)


        if mode == 'cell':
            
            p_cellHasText = self.config['p_cellhastext']
            cell_style = random.uniform(0,1)
            
            if cell_style <= (1-p_cellHasText): # very likely

                if rect[3]>rect[2]:
                    draw.text((rect[0]+5,rect[1]+5),text_heads[random.randint(0,len(text_heads)-1)],0,font=font)
                    num = self.gen_celltext(font_large, rect)
                    draw.text((rect[0]+5,rect[1]+rect[3]-line_height_large),num,0,font=font_large)
                else:
                    shift_up = random.randint(0,line_height_large//2+1)
                    idx = random.randint(0,len(text_heads)-1)
                    draw.text((rect[0]+5,rect[1]+rect[3]//2-line_height_large//2+shift_up),text_heads[idx],0,font=font_large)
                    num = self.gen_celltext(font_large, rect)
                    draw.text((rect[0]+10+font_large.getsize(text_heads[idx])[0],rect[1]+rect[3]//2-line_height_large//2+shift_up),num,0,font=font_large)
            else:
                
                text = lorem.paragraph()
                textlines = self.text_wrap(text, font, rect[2])

                line_height = font.getsize('glA')[1]*1.2
                for z in range(0,len(textlines)):
                    if 5+(z+1)*line_height > rect[3]:
                        break
                    draw.text((rect[0]+5,rect[1]+5+z*line_height),textlines[z],0,font=font)


        return im

    def draw_subcell_template(self, im, rect):

        draw = ImageDraw.Draw(im)
        min_h_split = 10
        min_w_split = 10

        #borders = ['full','horizontal','vertical']
        #border = borders[random.randint(0,2)]
        
        if random.uniform(0,1) < self.config['p_verticalsep_cells']:
            border = 'full'
        else:
            if random.uniform(0,1) < 0.5:
                border = 'horizontal'
            else:
                border = 'vertical'


        color = (random.randint(0,100)>90)*255

        width = random.randint(1,3)
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


    def split2(self, im, rect, lev, max_lev):

        area = 1000

        if lev > max_lev or random.randint(0,100) < 1: # probaility leaving big empty spaces
            im = self.rect_draw(im, rect)
            if random.random() > 0.5:
                im = self.put_text(im,rect,mode = 'cell')
            return im


        #im = rect_draw(im, rect)
        if lev > max_lev-1 and random.randint(0,100)<80:
            im = self.draw_subcell_template(im,rect)
            return im
        else:
            im = self.rect_draw(im, rect)
            #print(2)

        #if lev >2 or random.randint(0,100)<50:
        #    print(rect)
        #    print('^^^')
        #    im = draw_subcell_template(im,rect)
        #    return im

        #if lev > 2:
        #    return im

        #time.sleep(2)
        #im.show()

        if random.randint(0,100) < 50 and lev >= 2:
            # split vertically
            xcoord = random.randint(0,rect[2])
            xcoord = abs(int(np.random.normal(rect[2]//2,20)))
            #xcoord = rect[2]//2
            rect_left = (rect[0], rect[1], xcoord, rect[3]) 
            #im = rect_draw(im, rect_left)
            im = self.split2(im,rect_left,lev+1, max_lev) # left

            rect_right = (rect[0]+xcoord, rect[1], rect[2]-xcoord, rect[3])
            #im = rect_draw(im, rect_right)
            im = self.split2(im,rect_right,lev+1, max_lev) # right
        else:
            # split horizontally
            #ycoord = random.randint(0,rect[3])
            ycoord = abs(int(np.random.normal(rect[3]//2,20)))
            #ycoord = rect[3]//2
            rect_up = (rect[0], rect[1], rect[2], ycoord)  
            #print('up:')
            #print(rect_up)
            im = self.split2(im,rect_up,lev+1, max_lev) # left

            rect_down = (rect[0], rect[1]+ycoord, rect[2], rect[3]-ycoord)
            #print('down:')
            #print(rect_down)
            im = self.split2(im,rect_down,lev+1, max_lev) # right

        return im

    # -----------------------------------------------------------------------------------
    # ------------------------

    def make_table(self, im, rect, n_rows, n_cols, top=True):


        if top:
            cell_size = rect[2]//n_cols
            pad = 0
            keep_text = abs(random.normal(0,1))
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

        keep_text = abs(random.normal(0,1))
        for r in range(0,n_rows):
            if r == n_rows-1:
                    pad_height = rect[3]-cell_height*n_rows
            pad_width = 0

            set_text = False
            if keep_text > r/(n_rows-2):
                    set_text = True  

            for c in range(0, n_cols):
                if c == n_cols-1:
                    pad_width = rect[2]-cell_width*n_cols

                border = ('vertical' if random.uniform(0,1) < self.config['p_verticalsep_cells'] else 'full')
                
                if random.uniform(0,1) < self.config['p_verticalsep_cells']:
                    border = 'vertical'
                else:
                    if random.uniform(0,1) < 0.5:
                        border = 'full'
                    else:
                        border = 'horizontal'
                
                im = self.rect_draw(im, (rect[0]+c*cell_width,rect[1]+r*cell_height,cell_width+pad_width,cell_height+pad_height),border=border)

                if set_text:
                    im = self.put_text(im,(rect[0]+c*cell_width,rect[1]+r*cell_height,cell_width+pad_width,cell_height+pad_height),mode='tablecell')

        return im

    def calc_rc(self, rect):

        min_rows = 3
        min_cols = 2
        # at least 20px tall rows and 100px wide cols
        max_rows = rect[3]//20
        max_cols = rect[2]//100

        n_rows = max(min(max_rows,int(random.normal(self.config['avg_rows'],2))),min_rows)
        n_cols = max(min(max_cols,int(random.normal(self.config['avg_cols'],2))),min_cols)

        return n_rows, n_cols

    def show_image(self):
        assert(self.im is not None)
        self.im.show()

    def compose(self, im,rect):
        
        if self.config['template'] == 'bustapaga':

            im = self.rect_draw(im, rect)

            p_tableup = self.config['p_tableup']
            up_val = random.uniform(0.2,0.4)
            down_val = random.uniform(0.6,0.8)
            cut_up = int(random.normal(rect[3]*up_val,20))
            cut_down = int(random.normal(rect[3]*down_val,40))

            height_up = cut_up
            height_down = rect[3]-cut_down
            height_middle = rect[3]-height_down-height_up

            if random.random() < p_tableup:
                n_rows, n_cols = self.calc_rc(rect)
                im = self.make_table(im,(rect[0],rect[1],rect[2],height_up),n_rows,n_cols)
            else:
                im = self.split2(im,(rect[0],rect[1],rect[2],height_up),0,2)


            n_rows, n_cols = self.calc_rc(rect)
            im = self.make_table(im,(rect[0],rect[1]+height_up,rect[2],height_middle),n_rows, n_cols)


            im = self.split2(im,(rect[0],rect[1]+height_up+height_middle,rect[2],height_down),0,2)

            if self.scale != 1:
                im = im.resize((self.s_width, self.s_height))
                bg = Image.new("L", (self.width,self.height), 255)
                w1 = self.pad_scale_x//2
                h1 = self.pad_scale_y//2
                bg.paste(im, (w1, h1))
                self.im = bg
            else:
                self.im = im
            #self.im = spoil(im.convert("L"), options, crop = False)


# height = 3508
# width = 2479

# #height = 1400
# #width = 900

# borders = [100,100,100,200]
# im = Image.fromarray(255*np.ones((height,width)))

# tb = Tablegen(width,height)
# tb.compose(im,(0+borders[0],0+borders[2],width-(borders[0]+borders[1]),height-(borders[2]+borders[3])))
# tb.write_xml()
# tb.write_image()
# tb.show_image()