import pygame
import sys
#from pygame.locals import *
import json
import numpy as np
from FCL_classVer import FullConnectLayer as FCL
from CNN_classVer import ConvolutionLayer as CNN
#print (pygame.font.get_fonts())  # to get the name of different types of font pygame supports.
 
def save( filename, data_input ):
    print("Save as ", filename ,"...")
    data = {"x_input": data_input.tolist()   }
    f = open(filename, "w")
    json.dump(data, f)
    f.close()
class Panel(object):
    def __init__(self):
        pygame.init()
        self.font = pygame.font.SysFont("lucidafax", 25)
        self.font2 = pygame.font.SysFont("bodoni", 35)
        pygame.display.set_caption('Handwritten digit recognition demo')
        self.screen = pygame.display.set_mode((800,800), 0, 32)
        self.screen.fill(( (0,0,0)   ))
        self.draw_on=False
        self.color = (0, 0, 0)
        self.last_pos= (0,0)
        self.writing_penal_shape=(100,100,560,560)  
        self.start_ticks=pygame.time.get_ticks()  # to prevent double clicks the button.
        self.pixel_to_be_save=np.array([])
        self.which_Neural_Network = 'FullConnectLayer'
        pygame.display.update()
    def roundline(self,srf, color, start, end, radius=20):
        dx = end[0]-start[0]
        dy = end[1]-start[1]
        distance = max(abs(dx), abs(dy))
        for i in range(distance):
            x = int( start[0]+float(i)/distance*dx)
            y = int( start[1]+float(i)/distance*dy)
            if  self.check_inside_the_writing_penal(x,y):
                pygame.display.update(pygame.draw.circle(srf, color, (x, y), radius))


    def check_inside_the_writing_penal(self,x,y):  
        # 20 is the prevent the draw too close to the edge 
        if x > self.writing_penal_shape[0]+20 and x < self.writing_penal_shape[0]+self.writing_penal_shape[2]-20   \
            and  y > self.writing_penal_shape[1]+20 and y < self.writing_penal_shape[1]+self.writing_penal_shape[3]-20 :
                return True
        else :
            return False
    def remember_the_draw(self, reduce_size=20 ,monochrome=True ):
        pixel = np.array([])
        pixel_op = np.array([])
        for y in range(self.writing_penal_shape[1],self.writing_penal_shape[1]+self.writing_penal_shape[3] ): 
            for x in range(self.writing_penal_shape[0],self.writing_penal_shape[0]+self.writing_penal_shape[2] ): 
                if x % reduce_size ==0  and y %reduce_size == 0 and monochrome==True:              
                    color = self.screen.get_at( (x,y) )[0]
                    rearrange_pixel = [x/reduce_size - 5,y/reduce_size - 5,color/255]
                    pixel_op =np.append(pixel_op, np.array(  [1- (color/255) ] ))  
                    pixel =np.append(pixel, np.array(  [rearrange_pixel] ))  
        pixel = pixel.reshape(28,28,3)     
        print("append this pixel data into self.pixel_to_be_save ")
        self.pixel_to_be_save=np.append(self.pixel_to_be_save, np.array(  pixel_op )) 

    def save_into_json(self):
        save("test.json", self.pixel_to_be_save)
        
                
    def clean_up(self): 
        # Just cover the center-panel with a new rectangular
        self.rect =pygame.draw.rect(self.screen, ( (255,255,255) ), self.writing_penal_shape,0)
        pygame.display.update()
    def addRect(self):
        self.rect =pygame.draw.rect(self.screen, ( (255,255,255) ), self.writing_penal_shape, 0)
        pygame.display.update()

    def addText(self):
        
        self.screen.blit(self.font.render('Handwritten digit recognition demo! ', True, (255,255,255) ), (10, 10))
        self.screen.blit(self.font.render('Keyboard input: c=clean ; o=operate ; ESC=exit ' , True, (255,255,255) ),  (10, 35))
        self.screen.blit(self.font.render('Choose neural network : 1=FCL ; 2=CNN ; 3=Both   ' , True, (255,255,255) ),  (10, 60))

        pygame.display.update()
    def print_out_result(self,result) :    
        self.screen.blit(self.font2.render( "FCL Result : "+ str(result ) , True, (255,0,0) ), (110, 110))
        pygame.display.update()
    def print_out_result2(self,result) :    
        self.screen.blit(self.font2.render( "CNN Result : "+ str(result ) , True, (255,0,0) ), (420, 110))
        pygame.display.update()
    def print_out_used_NN(self):
        self.rect =pygame.draw.rect(self.screen, ( (0,0,0) ), (40,670,620,100),0)
        self.screen.blit(self.font.render( "Neural network : "+self.which_Neural_Network    , True, (255,255,255) ), (60, 700))
        pygame.display.update()

    def functionApp(self):
        if __name__ == '__main__':
            self.addRect()
            self.addText()
            self.print_out_used_NN()
            try :
                while True:
                    e = pygame.event.wait()
                    if e.type == pygame.QUIT:             raise StopIteration                    
                    if e.type == pygame.MOUSEBUTTONDOWN:  self.draw_on = True
                    if e.type == pygame.MOUSEBUTTONUP  :  self.draw_on = False
                    if e.type == pygame.MOUSEMOTION:
                        if self.draw_on:                            
                            self.roundline(self.screen, self.color, e.pos, self.last_pos)                          
                        self.last_pos = e.pos
                    keypressed = pygame.key.get_pressed()
                    if pygame.time.get_ticks() -  self.start_ticks > 500 :
                        if keypressed[pygame.K_1]:
                            self.which_Neural_Network = 'FullConnectLayer'
                            self.print_out_used_NN()                            
                        if keypressed[pygame.K_2]:
                            self.which_Neural_Network = 'ConvolutionLayer'
                            self.print_out_used_NN()
                        if keypressed[pygame.K_3]:
                            self.which_Neural_Network = 'Both'
                            self.print_out_used_NN()
                        if keypressed[pygame.K_c]:
                            print("clear up!")
                            self.clean_up()
                            self.start_ticks = pygame.time.get_ticks()
                        if keypressed[pygame.K_o]:
                            print("operate the handwritten digit recognization") 
                            self.remember_the_draw() 
                            self.save_into_json()
                            
                            if  self.which_Neural_Network == 'FullConnectLayer' :
                                result =FCL().predict_test() 
                            if  self.which_Neural_Network == 'ConvolutionLayer' :
                                result2  =CNN().predict_test()
                            if  self.which_Neural_Network == 'Both' :
                                result =FCL().predict_test() 
                                result2  =CNN().predict_test() 
                            self.clean_up()
                            self.start_ticks = pygame.time.get_ticks()
                            self.pixel_to_be_save=np.array([])
                            if self.which_Neural_Network == 'Both' or self.which_Neural_Network == 'FullConnectLayer' :
                                self.print_out_result(result) 
                            if self.which_Neural_Network == 'Both' or self.which_Neural_Network == 'ConvolutionLayer':
                                self.print_out_result2(result2) 
                            
                    if keypressed[pygame.K_ESCAPE]:
                        raise StopIteration

            except StopIteration:
                print("exit")
                pygame.display.quit() 
                pygame.quit()
                sys.exit()
                

 
display = Panel()
display.functionApp()
 
