#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 18:05:13 2021

@author: kevinm
"""
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
font = {'family' : 'serif',
#        'weight' : 'bold',
        'size'   : 16,}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import pearsonr, spearmanr, kendalltau


# top = tk.Tk()
# top.wm_title("Railcar Packing GUI")
# start_row = 2
# start_col = 1

# tk.Label(top,text='Initial Info',justify=tk.RIGHT).grid(row=start_row,column=start_col)

# possible_lengths = np.array([8,10,12,14,16,18,20])
# quantities = []
# for i,length in enumerate(possible_lengths):
#     tk.Label(top,text=f"{length}'").grid(row=start_row+1,column=start_col+i)
#     quantities.append(tk.Entry(top,width=10).grid(row=start_row+2,column=start_col+i))

# top.mainloop()

class railcar_properties(object):
    def __init__(self,weights,weight_errs):
        self.n_cars = 1 #number of railcars
        
        self.carlength = 575 #lineal footage of each railcar
        
        self.possible_lengths = np.array([8,10,12,14,16,18,20]) #possible board lengths
        # self.weights =          np.array([1, 2, 3, 4, 3, 2, 1]) #relative amounts of each type of board
        
        # self.weights = self.weights/np.sum(self.weights)
        # self.weight_errs = np.ones_like(self.weights)*0.02
        
        weight_sum = np.sum(weights)
        
        self.weights = weights/weight_sum #normalize weights
        self.weight_errs = weight_errs/weight_sum #normalize weight errors
        
        self.n_avg = self.carlength/np.sum(self.weights*self.possible_lengths)
        
        self.bins = np.zeros(len(self.possible_lengths)+1)
        self.bins[1:-1] = 0.5*(self.possible_lengths[1:]+self.possible_lengths[:-1])
        self.bins[0] = self.bins[1]-(self.possible_lengths[1]-self.possible_lengths[0])
        self.bins[-1] = self.bins[-2]+(self.possible_lengths[-1]-self.possible_lengths[-2])
        
        self.max_show = 5 #for printing and plotting
        self.max_keep = 100 #for saving outputs
        
        packing_outputs = self.calc_packing()
        self.best_dist_agree = packing_outputs[0]
        self.best_packing = packing_outputs[1]

                
    def calc_packing(self,printer=False,plotter=False):
        #use expectation to get amounts of each board length that fit in the car length
        n_avg = self.n_avg
        unrounded_quantities = n_avg*self.weights
        rounded_quantities = np.round(unrounded_quantities,0).astype(int) #round to integers for first guess
        
        final_quantities = np.copy(rounded_quantities)
        final_length = np.sum(rounded_quantities*self.possible_lengths)
        length_diff = self.carlength-final_length
        dist_diff = np.sum(np.abs(unrounded_quantities-final_quantities))
                
        if length_diff == 0: #perfect
            i = 0
            if printer:
                print('Perfect match to distribution and railcar length found!')
                print('\nOrder Num\t\t\t\t\t\t\t\t\t\tDiffFromCar\tDiffFromDist')
                print('--\t\tSize (ft):\t'+'\t'.join(self.possible_lengths.astype(str))+'\t--\t\t--')
                print(f'{i+1}\t\tQuantity:\t'+'\t'.join(final_quantities.astype(str))+f'\t{length_diff}\t\t{round(dist_diff,2)}')
            if plotter:
                plt.figure(figsize=(12,5))
                plt.hist(self.possible_lengths,weights=self.weights*n_avg,
                         bins=self.bins,histtype='step',lw=3,ls='--',label='Desired Distribution')
                plt.errorbar(self.possible_lengths,self.weights*n_avg,
                             yerr=self.weight_errs*n_avg,fmt='o',color='C0',capsize=5)
                plt.hist(self.possible_lengths,weights=final_quantities,
                         bins=self.bins+0.2*i,histtype='step',lw=2,ls='-',label='Order Num %d'%(i+1))
                plt.xlabel('Board Sizes (ft)'); plt.ylabel('Quantity')
                plt.grid(b=True, which='major', color='#666666', linestyle='-',alpha=0.3)
                plt.minorticks_on()
                plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.1)
                plt.legend(loc=6,bbox_to_anchor=(1.05,0.5))
                plt.tight_layout()
                plt.show()
                
            output_dist_sorting = np.zeros((1,len(self.possible_lengths)+2))
            output_packing_sorting = np.zeros((1,len(self.possible_lengths)+2))
            
            output_dist_sorting[0,:len(self.possible_lengths)] = final_quantities
            output_dist_sorting[0,len(self.possible_lengths)] = dist_diff
            output_packing_sorting[0,:len(self.possible_lengths)] = final_quantities
            output_packing_sorting[0,len(self.possible_lengths)] = dist_diff
        else:
            min_board_quantities = np.ceil(self.weights*n_avg-self.weight_errs*n_avg)
            max_board_quantities = np.floor(self.weights*n_avg+self.weight_errs*n_avg)   
            no_change = (final_quantities == 0) | (self.weight_errs == 0)
            min_board_quantities[no_change] = rounded_quantities[no_change]
            max_board_quantities[no_change] = rounded_quantities[no_change]
            
            max_change_quantities = max_board_quantities-min_board_quantities
            
            max_change_quantities_downup = np.zeros((2,len(max_change_quantities)))
            max_change_quantities_downup[:] = np.copy(max_change_quantities)
            max_change_quantities_downup[:,max_change_quantities < 1] = 1
            max_change_quantities_downup[0,(final_quantities-max_change_quantities) <= 0] = 0
            max_change_quantities_downup[:,no_change] = 0
            
            lowest_config = final_quantities-max_change_quantities_downup[0] #should always be lower than carlength
            changes_from_lowest = max_change_quantities_downup[1]+max_change_quantities_downup[0]    
            
            poss_changes = []
            for i in range(len(changes_from_lowest)):
                poss_changes.append(np.arange((changes_from_lowest[i]+1)))
            #this part needs to be updated if the number of possible lengths changes
            all_change_configs = np.meshgrid(poss_changes[0],poss_changes[1],poss_changes[2],poss_changes[3],
                                    poss_changes[4],poss_changes[5],poss_changes[6],indexing='ij',sparse=False)
            all_change_configs = np.array(all_change_configs).reshape((len(self.possible_lengths),-1)).T
            remaining_length = self.carlength-np.sum(lowest_config*self.possible_lengths)
            
            #for each change config, measure distance from distribution and from carlength
            dist_from_distribution = np.zeros(len(all_change_configs)) 
            dist_from_carlength = np.zeros(len(all_change_configs))
        
            for i in range(len(all_change_configs)):
                dist_from_carlength[i] = remaining_length-np.sum(all_change_configs[i]*self.possible_lengths)
                dist_from_distribution[i] = np.sum(np.abs(unrounded_quantities-(lowest_config+all_change_configs[i])))
                
            bad_packing = (dist_from_carlength < 0) #didn't fit in railcar
            dist_from_carlength[bad_packing] = np.inf
            dist_from_distribution[bad_packing] = np.inf
            
            best_packing_length = np.min(dist_from_carlength)
            best_packing_inds = (dist_from_carlength == best_packing_length)
            best_packing_configs = lowest_config+all_change_configs[best_packing_inds]
            #sort best packing configs by the distance from distribution
            sort_inds = np.argsort(dist_from_distribution[best_packing_inds])[:self.max_keep]
            best_packing_configs = best_packing_configs[sort_inds].astype(int)
            best_packing_config_dists = dist_from_distribution[best_packing_inds][sort_inds]
            
            n_save = len(sort_inds)
            
            output_packing_sorting = np.zeros((n_save,len(self.possible_lengths)+2))
            
            output_packing_sorting[:,:len(self.possible_lengths)] = best_packing_configs
            output_packing_sorting[:,len(self.possible_lengths)] = best_packing_config_dists
            output_packing_sorting[:,len(self.possible_lengths)+1] = best_packing_length
            
            best_dist_match = np.min(dist_from_distribution)
            best_dist_inds = (dist_from_distribution == best_dist_match)
            best_dist_configs = lowest_config+all_change_configs[best_dist_inds]
            #sort best distribution configs by the packing efficiency
            sort_inds = np.argsort(dist_from_carlength[best_dist_inds])[:self.max_keep]
            best_dist_configs = best_dist_configs[sort_inds].astype(int)
            best_dist_config_lengths = dist_from_carlength[best_dist_inds][sort_inds]
            
            n_save = len(sort_inds)
            
            output_dist_sorting = np.zeros((n_save,len(self.possible_lengths)+2))
            output_dist_sorting[:,:len(self.possible_lengths)] = best_dist_configs
            output_dist_sorting[:,len(self.possible_lengths)] = best_dist_match
            output_dist_sorting[:,len(self.possible_lengths)+1] = best_dist_config_lengths
            
            if printer:
                print('\nSORTING BY BEST MATCH TO DISTRIBUTION:')
                print('\nOrder Num\t\t\t\t\t\t\t\t\t\tDiffFromCar\tDiffFromDist')
                print('--\t\tSize (ft):\t'+'\t'.join(self.possible_lengths.astype(str))+'\t--\t\t--')
                for i in range(min(self.max_show,len(best_dist_configs))):
                    print(f'{i+1}\t\tQuantity:\t'+'\t'.join(best_dist_configs[i].astype(str))+f'\t{best_dist_config_lengths[i]}\t\t{round(best_dist_match,2)}')
            
            if plotter:
                plt.figure(figsize=(12,5))
                plt.hist(self.possible_lengths,weights=self.weights*n_avg,
                         bins=self.bins,histtype='step',lw=3,ls='--',label='Desired Distribution')
                plt.errorbar(self.possible_lengths,self.weights*n_avg,
                             yerr=self.weight_errs*n_avg,fmt='o',color='C0',capsize=5)
                for i in range(min(self.max_show,len(best_dist_configs))):
                    plt.hist(self.possible_lengths,weights=best_dist_configs[i],
                             bins=self.bins+0.2*i,histtype='step',lw=2,
                             ls='-',label='Order Num %d'%(i+1))
                plt.xlabel('Board Sizes (ft)'); plt.ylabel('Quantity')
                plt.grid(b=True, which='major', color='#666666', linestyle='-',alpha=0.3)
                plt.minorticks_on()
                plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.1)
                plt.legend(loc=6,bbox_to_anchor=(1.05,0.5))
                plt.tight_layout()
                plt.show()
                
            if printer:
            
                print('\n\nSORTING BY BEST RAILCAR PACKING:')
                print('\nOrder Num\t\t\t\t\t\t\t\t\t\tDiffFromCar\tDiffFromDist')
                print('--\t\tSize (ft):\t'+'\t'.join(self.possible_lengths.astype(str))+'\t--\t\t--')
                for i in range(min(self.max_show,len(best_packing_configs))):
                    print(f'{i+1}\t\tQuantity:\t'+'\t'.join(best_packing_configs[i].astype(str))+f'\t{best_packing_length}\t\t{round(best_packing_config_dists[i],2)}')
        
            if plotter:
                plt.figure(figsize=(12,5))
                plt.hist(self.possible_lengths,weights=self.weights*n_avg,
                         bins=self.bins,histtype='step',lw=3,ls='--',label='Desired Distribution')
                plt.errorbar(self.possible_lengths,self.weights*n_avg,
                             yerr=self.weight_errs*n_avg,fmt='o',color='C0',capsize=5)
                for i in range(min(self.max_show,len(best_packing_configs))):
                    plt.hist(self.possible_lengths,weights=best_packing_configs[i],
                             bins=self.bins+0.2*i,histtype='step',
                             lw=2,ls='-',label='Order Num %d'%(i+1))
                plt.xlabel('Board Sizes (ft)'); plt.ylabel('Quantity')
                plt.grid(b=True, which='major', color='#666666', linestyle='-',alpha=0.3)
                plt.minorticks_on()
                plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.1)
                plt.legend(loc=6,bbox_to_anchor=(1.05,0.5))
                plt.tight_layout()
                plt.show()
                
        return output_dist_sorting,output_packing_sorting

#ratio of weights for       [8  10 12 14 16 18 20] foot lengths
weights =          np.array([1, 2, 3, 4, 3, 2, 1]) #relative amounts of each type of board
weight_errs =      np.ones_like(weights)*0.5
# weights =          np.array([3, 5, 8, 10, 8, 5, 3]) #relative amounts of each type of board
# weight_errs =      np.ones_like(weights)*0.5

# test = railcar_properties(weights,weight_errs)


matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
NavigationToolbar2Tk)
from matplotlib.figure import Figure


class Scrollable(tk.Frame):
    """
       Make a frame scrollable with scrollbar on the right.
       After adding or removing widgets to the scrollable frame, 
       call the update() method to refresh the scrollable area.
    """

    def __init__(self, frame, width=16):

        scrollbar = tk.Scrollbar(frame, width=width)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, expand=False)

        self.canvas = tk.Canvas(frame, yscrollcommand=scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar.config(command=self.canvas.yview)

        self.canvas.bind('<Configure>', self.__fill_canvas)

        # base class initialization
        tk.Frame.__init__(self, frame)         

        # assign this obj (the inner frame) to the windows item of the canvas
        self.windows_item = self.canvas.create_window(0,0, window=self, anchor=tk.NW)


    def __fill_canvas(self, event):
        "Enlarge the windows item to the canvas width"

        canvas_width = event.width
        self.canvas.itemconfig(self.windows_item, width = canvas_width)        

    def update(self):
        "Update the canvas and the scrollregion"

        self.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox(self.windows_item))

class App:
    def __init__(self, master):
        #initial ratio of weights for [8  10 12 14 16 18 20] foot lengths
        weights =            np.array([3, 5, 8, 10, 8, 5, 3]) #relative amounts of each type of board
        weight_errs =        np.ones_like(weights)
        
        start_row = 0
        start_col = 0
        
        self.railcar_props = railcar_properties(weights,weight_errs)
        self.max_display = 20 #max number of entries into each table
        
        curr_ratios = np.copy(weights)
        curr_tols = np.copy(weight_errs)
        # create a container for buttons
		
		# create container for text
        textFrame = tk.Frame(root)
        textFrame.grid(row=start_row,column=start_col)
        start_row += 1
        
		# create text        
        tk.Label(textFrame,
                 text='Input relative ratios of product to determine distribution.',
                 justify=tk.RIGHT).grid(row=start_row,column=1,columnspan=6)
        
        possible_lengths = self.railcar_props.possible_lengths
        self.ratios = [] #ratios of wood dimension quantities
        self.tolerances = [] #quantity tolerances (i.e. 1 means +/- 1)
        tk.Label(textFrame,text='Dimension:',justify=tk.LEFT).grid(row=start_row+1,column=0)
        for i,length in enumerate(possible_lengths):
            tk.Label(textFrame,text=f"{length}'").grid(row=start_row+1,column=i+1)
        tk.Label(textFrame,text='Quantity Ratio:',justify=tk.LEFT).grid(row=start_row+2,column=0)
        for i,length in enumerate(possible_lengths):
            quantity_box = tk.Entry(textFrame,width=5)
            quantity_box.insert(0,str(curr_ratios[i]))
            quantity_box.grid(row=start_row+2,column=i+1)
            self.ratios.append(quantity_box)
        tk.Label(textFrame,text='Quantity Tolerance:',justify=tk.LEFT).grid(row=start_row+3,column=0)
        for i,length in enumerate(possible_lengths):
            tol_box = tk.Entry(textFrame,width=5)
            tol_box.insert(0,str(curr_tols[i]))
            tol_box.grid(row=start_row+3,column=i+1)
            self.tolerances.append(tol_box)
            
        start_row += 5
        
        buttonFrame = tk.Frame(root)
        buttonFrame.grid(row=start_row,column=start_col)
		
		# create buttons
        self.buttonGenerate = tk.Button(master=buttonFrame,
						           text='Update',
						           command=self.update_weights)
        self.buttonGenerate.grid(row=start_row,column=start_col+6)
        self.buttonQuit = tk.Button(master=buttonFrame,
					           text='Quit',
					           command=root.destroy)
        self.buttonQuit.grid(row=start_row,column=start_col+7)
        
        start_row += 2
		      
        tabControl = ttk.Notebook(root)
          
        tab_best_pack = ttk.Frame(tabControl)
        tab_best_dist = ttk.Frame(tabControl)
        tab_curr_dist = ttk.Frame(tabControl)
          
        tabControl.add(tab_curr_dist, text ='Current Distribution')
        tabControl.add(tab_best_pack, text ='Best Packing')
        tabControl.add(tab_best_dist, text ='Best Distribution')
        tabControl.grid(row=start_row,column=0)
                
        best_dist_agree = self.railcar_props.best_dist_agree 
        best_dist_configs = best_dist_agree[:,:len(possible_lengths)]
        best_dist_config_dists = best_dist_agree[:,len(possible_lengths)]
        best_dist_config_lengths = best_dist_agree[:,len(possible_lengths)+1]
        best_packing = self.railcar_props.best_packing 
        best_packing_configs = best_packing[:,:len(possible_lengths)]
        best_packing_config_dists = best_packing[:,len(possible_lengths)]
        best_packing_config_lengths = best_packing[:,len(possible_lengths)+1]
        
        start_row += 1
        
        tk.Label(tab_best_pack,text='Plot',justify=tk.LEFT).grid(row=start_row,column=0)
        tk.Label(tab_best_pack,text='Order Number',justify=tk.LEFT).grid(row=start_row,column=1)
        tk.Label(tab_best_pack,text='Quantity of Dimension',justify=tk.LEFT).grid(row=start_row,column=2,columnspan=len(possible_lengths))
        tk.Label(tab_best_pack,text='Distribution Diff.',justify=tk.LEFT).grid(row=start_row,column=3+len(possible_lengths))
        tk.Label(tab_best_pack,text=r'Length Diff.',justify=tk.LEFT).grid(row=start_row,column=4+len(possible_lengths))
        for i,length in enumerate(possible_lengths):
            tk.Label(tab_best_pack,text=f"{length}'").grid(row=start_row+1,column=i+2)
            
        start_row += 2
        
        tk.Label(tab_best_dist,text='Plot',justify=tk.LEFT).grid(row=start_row,column=0)
        tk.Label(tab_best_dist,text='Order Number',justify=tk.LEFT).grid(row=start_row,column=1)
        tk.Label(tab_best_dist,text='Quantity of Dimension',justify=tk.LEFT).grid(row=start_row,column=2,columnspan=len(possible_lengths))
        tk.Label(tab_best_dist,text='Distribution Diff.',justify=tk.LEFT).grid(row=start_row,column=3+len(possible_lengths))
        tk.Label(tab_best_dist,text=r'Length Diff.',justify=tk.LEFT).grid(row=start_row,column=4+len(possible_lengths))
        for i,length in enumerate(possible_lengths):
            tk.Label(tab_best_dist,text=f"{length}'").grid(row=start_row+1,column=i+2)

        start_row += 2
        
        self.pack_text_frame = ttk.Frame(tab_best_pack)
        self.pack_text_frame.grid(row=start_row,column=0,columnspan=12,sticky='nsew')
        self.scrollable_body_pack = Scrollable(self.pack_text_frame, width=16)
        
        self.dist_text_frame = ttk.Frame(tab_best_dist)
        self.dist_text_frame.grid(row=start_row,column=0,columnspan=12,sticky='nsew')
        self.scrollable_body_dist = Scrollable(self.dist_text_frame, width=16)        
        
        self.list_space_nums = [12,13,14,14,13,13,13]
        self.no_list_space_nums = [14,13,14,14,14,14,13]
        
        self.ignore_first_change = True #don't replot while generating buttons
        
        if len(best_packing_configs) >= 10:
            space_nums = self.list_space_nums
        else:
            space_nums = self.no_list_space_nums
        
        n_display = min(self.max_display,len(best_packing_configs))
        self.pack_checkboxes = []
        self.pack_checkbox_states = []
        for i in range(n_display):
            button = ttk.Checkbutton(self.scrollable_body_pack,text='\t'+str(i+1),command=self.update_pack_plot)
            button.invoke()
            if i != 0:
                button.invoke()
                self.pack_checkbox_states.append(0)
            else:
                self.pack_checkbox_states.append(1)
            button.grid(row=i,column=0,sticky='nsew')
            for j in range(len(best_packing_configs[i])):
                string = str(int(best_packing_configs[i,j]))
                n_space = space_nums[j]+1-len(string)
                if j == 0:
                    string = '\t'+' '*n_space+string
                else:
                    string = ' '*n_space+string
                curr_button_label = ttk.Label(self.scrollable_body_pack,text=string)
                curr_button_label.grid(row=i,column=j+1,sticky='nsew')
            self.pack_checkboxes.append(button)
            j += 1
            curr_button_label = ttk.Label(self.scrollable_body_pack,text='\t\t%.02f'%(round(best_packing_config_dists[i],2)))
            curr_button_label.grid(row=i,column=j+1,sticky='nsew')
            j += 1
            curr_button_label = ttk.Label(self.scrollable_body_pack,text='\t%.02f'%(round(best_packing_config_lengths[i],2)))
            curr_button_label.grid(row=i,column=j+1,sticky='nsew')
        self.scrollable_body_pack.update()        
                
        
        if len(best_dist_configs) >= 10:
            space_nums = self.list_space_nums
        else:
            space_nums = self.no_list_space_nums
        
        n_display = min(self.max_display,len(best_dist_configs))
        self.dist_checkboxes = []
        self.dist_checkbox_states = []
        for i in range(n_display):
            button = ttk.Checkbutton(self.scrollable_body_dist,text='\t'+str(i+1),command=self.update_dist_plot)
            button.invoke()
            if i != 0:
                button.invoke()
                self.dist_checkbox_states.append(0)
            else:
                self.dist_checkbox_states.append(1)
            button.grid(row=i,column=0,sticky='nsew')
            for j in range(len(best_dist_configs[i])):
                string = str(int(best_dist_configs[i,j]))
                n_space = space_nums[j]+1-len(string)
                if j == 0:
                    string = '\t'+' '*n_space+string
                else:
                    string = ' '*n_space+string
                curr_button_label = ttk.Label(self.scrollable_body_dist,text=string)
                curr_button_label.grid(row=i,column=j+1,sticky='nsew')
            self.dist_checkboxes.append(button)
            j += 1
            curr_button_label = ttk.Label(self.scrollable_body_dist,text='\t\t%.02f'%(round(best_dist_config_dists[i],2)))
            curr_button_label.grid(row=i,column=j+1,sticky='nsew')
            j += 1
            curr_button_label = ttk.Label(self.scrollable_body_dist,text='\t%.02f'%(round(best_dist_config_lengths[i],2)))
            curr_button_label.grid(row=i,column=j+1,sticky='nsew')
        self.scrollable_body_dist.update()   
        
        self.ignore_first_change = False #allow buttons to work now
                
        start_row += 1
		
        figsize = [10,4]
		# create plot
        self.f_tab_pack = Figure(figsize=figsize, dpi=100)
        self.ax_tab_pack = self.f_tab_pack.add_subplot(111)
        # self.ax_tab1.set_xlim([7, 21])
        # self.ax_tab1.set_ylim([0, 20])
		
        self.canvas_tab_pack = FigureCanvasTkAgg(self.f_tab_pack, master=tab_best_pack)
        self.canvas_tab_pack.draw()
        self.canvas_tab_pack.get_tk_widget().grid(row=start_row,column=0,columnspan=12)

        self.f_tab_curr_dist = Figure(figsize=figsize, dpi=100)
        self.ax_tab_curr_dist = self.f_tab_curr_dist.add_subplot(111)
        # self.ax_tab2.set_xlim([7, 21])
        # self.ax_tab2.set_ylim([0, 20])
		
        self.canvas_tab_curr_dist = FigureCanvasTkAgg(self.f_tab_curr_dist, master=tab_curr_dist)
        self.canvas_tab_curr_dist.draw()
        self.canvas_tab_curr_dist.get_tk_widget().grid(row=start_row,column=0,columnspan=12)

        self.f_tab_best_dist = Figure(figsize=figsize, dpi=100)
        self.ax_tab_best_dist = self.f_tab_best_dist.add_subplot(111)
        # self.ax_tab2.set_xlim([7, 21])
        # self.ax_tab2.set_ylim([0, 20])
		
        self.canvas_tab_best_dist = FigureCanvasTkAgg(self.f_tab_best_dist, master=tab_best_dist)
        self.canvas_tab_best_dist.draw()
        self.canvas_tab_best_dist.get_tk_widget().grid(row=start_row,column=0,columnspan=12)
                
        self.railcarDist_plotter()
	
    def generateMap(self):
		# generate random line
        c = np.random.rand()
        m = np.random.rand() - c
		
		# get data points
        pointCnt = 50
        sigma = np.random.rand()
        x = np.random.rand(pointCnt)
        y = m*x + c + sigma * np.random.randn(pointCnt)
		
		# update text
        corr = pearsonr(x,y)[0]
        rho  = spearmanr(x,y)[0]
        tau  = kendalltau(x,y)[0]
        newVals = """Pearson correlation:\t%.2f\nSpearman rho:\t%.2f\nKendall tau:\t%.2f""" % (corr, rho, tau)
        self.label.config(text=newVals)
		
		# plot points
        self.ax.clear()
        self.ax.scatter(x, y, marker='s', c='black')
        self.ax.set_xlim([-0.2, 1.2])
        self.canvas.draw()

    def railcarDist_plotter(self,packing_inds=[0],dist_inds=[0]):
        
        possible_lengths = self.railcar_props.possible_lengths
        weights = self.railcar_props.weights
        weight_errs = self.railcar_props.weight_errs
        n_avg = self.railcar_props.n_avg
        bins = self.railcar_props.bins
        max_show = self.railcar_props.max_show
        
        best_dist_agree = self.railcar_props.best_dist_agree 
        best_dist_configs = best_dist_agree[:,:len(possible_lengths)]
        best_packing = self.railcar_props.best_packing 
        best_packing_configs = best_packing[:,:len(possible_lengths)]
                
        self.ax_tab_pack.clear()
        self.ax_tab_pack.hist(possible_lengths,weights=weights*n_avg,
                 bins=bins,histtype='step',lw=3,ls='--',label='Desired Distribution',zorder=1e10,color='k')
        self.ax_tab_pack.errorbar(possible_lengths,weights*n_avg,
                     yerr=weight_errs*n_avg,fmt='o',color='k',capsize=5,zorder=1e10)
        # for i in range(min(max_show,len(best_packing_configs))):
        for i,ind in enumerate(packing_inds):
            self.ax_tab_pack.hist(possible_lengths,weights=best_packing_configs[ind],
                     bins=bins+0.2*i,label='Order Number %d'%(ind+1))
        self.ax_tab_pack.set_xticks(possible_lengths)
        ylim = self.ax_tab_pack.get_ylim()
        yticks = self.ax_tab_pack.get_yticks()
        new_yticks = np.arange(0,yticks.max()+1,2).astype(int)
        new_mintor_yticks = np.arange(1,yticks.max()+1,2).astype(int)
        self.ax_tab_pack.set_yticks(new_yticks,minor=False)
        self.ax_tab_pack.set_yticks(new_mintor_yticks,minor=True)
        self.ax_tab_pack.set_ylim(ylim)
        self.ax_tab_pack.set_xlabel('Board Sizes (ft)')
        self.ax_tab_pack.set_ylabel('Quantity per Railcar')
        self.ax_tab_pack.grid(b=True, which='major', color='#666666', linestyle='-',alpha=0.3)
        self.ax_tab_pack.xaxis.set_tick_params(which='minor', bottom=False)
        self.ax_tab_pack.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.1)
        self.ax_tab_pack.legend(loc=6,bbox_to_anchor=(1.05,0.5),fontsize=10)
        self.f_tab_pack.tight_layout()
        self.canvas_tab_pack.draw()
        
        self.ax_tab_best_dist.clear()
        self.ax_tab_best_dist.hist(possible_lengths,weights=weights*n_avg,
                 bins=bins,histtype='step',lw=3,ls='--',label='Desired Distribution',zorder=1e10,color='k')
        self.ax_tab_best_dist.errorbar(possible_lengths,weights*n_avg,
                     yerr=weight_errs*n_avg,fmt='o',color='k',capsize=5,zorder=1e10)
        # for i in range(min(max_show,len(best_dist_configs))):
        for i,ind in enumerate(dist_inds):
            self.ax_tab_best_dist.hist(possible_lengths,weights=best_dist_configs[ind],
                     bins=bins+0.2*i,label='Order Number %d'%(ind+1))
        self.ax_tab_best_dist.set_xticks(possible_lengths)
        ylim = self.ax_tab_best_dist.get_ylim()
        yticks = self.ax_tab_best_dist.get_yticks()
        new_yticks = np.arange(0,yticks.max()+1,2).astype(int)
        new_mintor_yticks = np.arange(1,yticks.max()+1,2).astype(int)
        self.ax_tab_best_dist.set_yticks(new_yticks,minor=False)
        self.ax_tab_best_dist.set_yticks(new_mintor_yticks,minor=True)
        self.ax_tab_best_dist.set_ylim(ylim)
        self.ax_tab_best_dist.set_xlabel('Board Sizes (ft)')
        self.ax_tab_best_dist.set_ylabel('Quantity per Railcar')
        self.ax_tab_best_dist.grid(b=True, which='major', color='#666666', linestyle='-',alpha=0.3)
        self.ax_tab_best_dist.xaxis.set_tick_params(which='minor', bottom=False)
        self.ax_tab_best_dist.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.1)
        self.ax_tab_best_dist.legend(loc=6,bbox_to_anchor=(1.05,0.5),fontsize=10)
        self.f_tab_best_dist.tight_layout()
        self.canvas_tab_best_dist.draw()

        self.ax_tab_curr_dist.clear()
        self.ax_tab_curr_dist.hist(possible_lengths,weights=weights*n_avg,
                 bins=bins,histtype='step',lw=3,ls='-',label='Desired Distribution',color='k')
        self.ax_tab_curr_dist.errorbar(possible_lengths,weights*n_avg,
                     yerr=weight_errs*n_avg,fmt='o',color='k',capsize=5)
        self.ax_tab_curr_dist.set_xticks(possible_lengths)
        ylim = self.ax_tab_curr_dist.get_ylim()
        yticks = self.ax_tab_curr_dist.get_yticks()
        new_yticks = np.arange(0,yticks.max()+1,2).astype(int)
        new_mintor_yticks = np.arange(1,yticks.max()+1,2).astype(int)
        self.ax_tab_curr_dist.set_yticks(new_yticks,minor=False)
        self.ax_tab_curr_dist.set_yticks(new_mintor_yticks,minor=True)
        self.ax_tab_curr_dist.set_ylim(ylim)
        self.ax_tab_curr_dist.set_xlabel('Board Sizes (ft)')
        self.ax_tab_curr_dist.set_ylabel('Quantity per Railcar')
        self.ax_tab_curr_dist.grid(b=True, which='major', color='#666666', linestyle='-',alpha=0.3)
        self.ax_tab_curr_dist.xaxis.set_tick_params(which='minor', bottom=False)
        self.ax_tab_curr_dist.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.1)
        self.f_tab_curr_dist.tight_layout()
        self.canvas_tab_curr_dist.draw()
        
        
    def update_weights(self):
        new_weights = np.zeros(len(self.railcar_props.possible_lengths))
        new_tols = np.zeros(len(self.railcar_props.possible_lengths))
        
        for i in range(len(self.ratios)):
            new_weights[i] = float(self.ratios[i].get())
            new_tols[i] = float(self.tolerances[i].get())
                    
        self.railcar_props = railcar_properties(new_weights,new_tols)
        self.railcarDist_plotter()
        
        possible_lengths = self.railcar_props.possible_lengths
        
        best_dist_agree = self.railcar_props.best_dist_agree 
        best_dist_configs = best_dist_agree[:,:len(possible_lengths)]
        best_dist_config_dists = best_dist_agree[:,len(possible_lengths)]
        best_dist_config_lengths = best_dist_agree[:,len(possible_lengths)+1]
        best_packing = self.railcar_props.best_packing 
        best_packing_configs = best_packing[:,:len(possible_lengths)]
        best_packing_config_dists = best_packing[:,len(possible_lengths)]
        best_packing_config_lengths = best_packing[:,len(possible_lengths)+1]
        
        if len(best_packing_configs) >= 10:
            space_nums = self.list_space_nums
        else:
            space_nums = self.no_list_space_nums
            
        self.ignore_first_change = True #allow buttons to work now
            
        n_display = min(self.max_display,len(best_packing_configs))
        self.pack_checkboxes = []
        self.pack_checkbox_states = []
        for i in range(n_display):
            button = ttk.Checkbutton(self.scrollable_body_pack,text='\t'+str(i+1),command=self.update_pack_plot)
            button.invoke()
            if i != 0:
                button.invoke()
                self.pack_checkbox_states.append(0)
            else:
                self.pack_checkbox_states.append(1)
            button.grid(row=i,column=0,sticky='nsew')
            for j in range(len(best_packing_configs[i])):
                string = str(int(best_packing_configs[i,j]))
                n_space = space_nums[j]+1-len(string)
                if j == 0:
                    string = '\t'+' '*n_space+string
                else:
                    string = ' '*n_space+string
                curr_button_label = ttk.Label(self.scrollable_body_pack,text=string)
                curr_button_label.grid(row=i,column=j+1,sticky='nsew')
            self.pack_checkboxes.append(button)
            j += 1
            curr_button_label = ttk.Label(self.scrollable_body_pack,text='\t\t%.02f'%(round(best_packing_config_dists[i],2)))
            curr_button_label.grid(row=i,column=j+1,sticky='nsew')
            j += 1
            curr_button_label = ttk.Label(self.scrollable_body_pack,text='\t%.02f'%(round(best_packing_config_lengths[i],2)))
            curr_button_label.grid(row=i,column=j+1,sticky='nsew')
        self.scrollable_body_pack.update()        
                
        if len(best_dist_configs) >= 10:
            space_nums = self.list_space_nums
        else:
            space_nums = self.no_list_space_nums
            
        n_display = min(self.max_display,len(best_dist_configs))
        self.dist_checkboxes = []
        self.dist_checkbox_states = []
        for i in range(n_display):
            button = ttk.Checkbutton(self.scrollable_body_dist,text='\t'+str(i+1),command=self.update_dist_plot)
            button.invoke()
            if i != 0:
                button.invoke()
                self.dist_checkbox_states.append(0)
            else:
                self.dist_checkbox_states.append(1)
            button.grid(row=i,column=0,sticky='nsew')
            for j in range(len(best_dist_configs[i])):
                string = str(int(best_dist_configs[i,j]))
                n_space = space_nums[j]+1-len(string)
                if j == 0:
                    string = '\t'+' '*n_space+string
                else:
                    string = ' '*n_space+string
                curr_button_label = ttk.Label(self.scrollable_body_dist,text=string)
                curr_button_label.grid(row=i,column=j+1,sticky='nsew')
            self.dist_checkboxes.append(button)
            j += 1
            curr_button_label = ttk.Label(self.scrollable_body_dist,text='\t\t%.02f'%(round(best_dist_config_dists[i],2)))
            curr_button_label.grid(row=i,column=j+1,sticky='nsew')
            j += 1
            curr_button_label = ttk.Label(self.scrollable_body_dist,text='\t%.02f'%(round(best_dist_config_lengths[i],2)))
            curr_button_label.grid(row=i,column=j+1,sticky='nsew')
        self.scrollable_body_dist.update()       
        
        self.ignore_first_change = False #allow buttons to work now
        
    def update_pack_plot(self):
        if self.ignore_first_change:
            pass
        else:
            #run this when a checkbox has been ticked, loop over the 
            #boxes to determine which to plot, then plot the highlighted 
            
            states = np.zeros(len(self.pack_checkboxes)).astype(bool)
            for i,box in enumerate(self.pack_checkboxes):
                if 'selected' in box.state():
                    states[i] = True            
            inds = np.where(states)[0]
            
            possible_lengths = self.railcar_props.possible_lengths
            weights = self.railcar_props.weights
            weight_errs = self.railcar_props.weight_errs
            n_avg = self.railcar_props.n_avg
            bins = self.railcar_props.bins
            n_bins = len(bins)-1
            dbins = bins[1]-bins[0]
            
            if len(inds) == 1:
                new_bins = np.copy(bins)
            else:
                new_bins = np.zeros(n_bins*len(inds)+1)
                new_bins[0] = bins[0]
                new_bins[-1] = bins[-1]
                for i in range(len(inds)):
                    new_bins[i:-1:len(inds)] = bins[:-1]+i/len(inds)*dbins
            new_centers = 0.5*(new_bins[1:]+new_bins[:-1])
            
            best_packing = self.railcar_props.best_packing 
            best_packing_configs = best_packing[:,:len(possible_lengths)]
            
            self.ax_tab_pack.clear()
            # self.ax_tab1.set_xlim([7, 21])
            # self.ax_tab1.set_ylim([0, 20])
		            
            self.ax_tab_pack.hist(possible_lengths,weights=weights*n_avg,
                     bins=bins,histtype='step',lw=3,ls='--',label='Desired Distribution',zorder=1e10,color='k')
            self.ax_tab_pack.errorbar(possible_lengths,weights*n_avg,
                         yerr=weight_errs*n_avg,fmt='o',color='k',capsize=5,zorder=1e10)
            # for i in range(min(max_show,len(best_packing_configs))):
            for i,ind in enumerate(inds):
                new_weights = np.zeros(len(new_bins)-1)
                new_weights[i::len(inds)] = best_packing_configs[ind]
                self.ax_tab_pack.hist(new_centers,weights=new_weights,
                         bins=new_bins,label='Order Number %d'%(ind+1))
            self.ax_tab_pack.set_xticks(possible_lengths)
            ylim = self.ax_tab_pack.get_ylim()
            yticks = self.ax_tab_pack.get_yticks()
            new_yticks = np.arange(0,yticks.max()+1,2).astype(int)
            new_mintor_yticks = np.arange(1,yticks.max()+1,2).astype(int)
            self.ax_tab_pack.set_yticks(new_yticks,minor=False)
            self.ax_tab_pack.set_yticks(new_mintor_yticks,minor=True)
            self.ax_tab_pack.set_ylim(ylim)
            self.ax_tab_pack.set_xlabel('Board Sizes (ft)')
            self.ax_tab_pack.set_ylabel('Quantity per Railcar')
            self.ax_tab_pack.grid(b=True, which='major', color='#666666', linestyle='-',alpha=0.3)
            self.ax_tab_pack.xaxis.set_tick_params(which='minor', bottom=False)
            self.ax_tab_pack.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.1)
            self.ax_tab_pack.legend(loc=6,bbox_to_anchor=(1.05,0.5),fontsize=10)
            self.f_tab_pack.tight_layout()
            self.canvas_tab_pack.draw()

    def update_dist_plot(self):
        if self.ignore_first_change:
            pass
        else:
            #run this when a checkbox has been ticked, loop over the 
            #boxes to determine which to plot, then plot the highlighted 
            
            states = np.zeros(len(self.dist_checkboxes)).astype(bool)
            for i,box in enumerate(self.dist_checkboxes):
                if 'selected' in box.state():
                    states[i] = True            
            inds = np.where(states)[0]
            
            possible_lengths = self.railcar_props.possible_lengths
            weights = self.railcar_props.weights
            weight_errs = self.railcar_props.weight_errs
            n_avg = self.railcar_props.n_avg
            bins = self.railcar_props.bins
            n_bins = len(bins)-1
            dbins = bins[1]-bins[0]
            
            if len(inds) == 1:
                new_bins = np.copy(bins)
            else:
                new_bins = np.zeros(n_bins*len(inds)+1)
                new_bins[0] = bins[0]
                new_bins[-1] = bins[-1]
                for i in range(len(inds)):
                    new_bins[i:-1:len(inds)] = bins[:-1]+i/len(inds)*dbins
            new_centers = 0.5*(new_bins[1:]+new_bins[:-1])
            
            best_dist_agree = self.railcar_props.best_dist_agree 
            best_dist_configs = best_dist_agree[:,:len(possible_lengths)]
            
            self.ax_tab_best_dist.clear()
            # self.ax_tab1.set_xlim([7, 21])
            # self.ax_tab1.set_ylim([0, 20])
		            
            self.ax_tab_best_dist.hist(possible_lengths,weights=weights*n_avg,
                     bins=bins,histtype='step',lw=3,ls='--',label='Desired Distribution',zorder=1e10,color='k')
            self.ax_tab_best_dist.errorbar(possible_lengths,weights*n_avg,
                         yerr=weight_errs*n_avg,fmt='o',color='k',capsize=5,zorder=1e10)
            for i,ind in enumerate(inds):
                new_weights = np.zeros(len(new_bins)-1)
                new_weights[i::len(inds)] = best_dist_configs[ind]
                self.ax_tab_best_dist.hist(new_centers,weights=new_weights,
                         bins=new_bins,label='Order Number %d'%(ind+1))
            self.ax_tab_best_dist.set_xticks(possible_lengths)
            ylim = self.ax_tab_best_dist.get_ylim()
            yticks = self.ax_tab_best_dist.get_yticks()
            new_yticks = np.arange(0,yticks.max()+1,2).astype(int)
            new_mintor_yticks = np.arange(1,yticks.max()+1,2).astype(int)
            self.ax_tab_best_dist.set_yticks(new_yticks,minor=False)
            self.ax_tab_best_dist.set_yticks(new_mintor_yticks,minor=True)
            self.ax_tab_best_dist.set_ylim(ylim)
            self.ax_tab_best_dist.set_xlabel('Board Sizes (ft)')
            self.ax_tab_best_dist.set_ylabel('Quantity per Railcar')
            self.ax_tab_best_dist.grid(b=True, which='major', color='#666666', linestyle='-',alpha=0.3)
            self.ax_tab_best_dist.xaxis.set_tick_params(which='minor', bottom=False)
            self.ax_tab_best_dist.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.1)
            self.ax_tab_best_dist.legend(loc=6,bbox_to_anchor=(1.05,0.5),fontsize=10)
            self.f_tab_best_dist.tight_layout()
            self.canvas_tab_best_dist.draw()
        
        
root = tk.Tk()
root.title("Railcar Packing")
app = App(root)
root.mainloop()





