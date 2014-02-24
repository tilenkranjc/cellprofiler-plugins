'''<b>Measure RWC</b> measures the Rank Weighted Correlation (RWC) between intensities in different images (e.g.,
different color channels) on a pixel-by-pixel basis, within identified
objects or across an entire image
<hr>

Given two or more images, this module calculates the RWC between the
pixel intensities. The RWC can be measured for entire
images, or within each individual object.

RWC was developed by Dr. Vasanth R. Singan. Please cite the following paper:
Singan VR, Jones TR, Curran KM, Simpson JC. Dual channel rank-based intensity 
weighting for quantitative co-localization of microscopy images. 
BMC Bioinformatics. 2011;12:407.

<h4>Available measurements</h4>
<ul>
<li><i>RWC I over J:</i> The RWC of image I over image J. </li>
<li><i>RWC J over I:</i> The RWC of image J over image I. </li>
</ul>

RWCs will be calculated between all pairs of images that are selected in 
the module, as well as between selected objects. For example, if RWCs 
are to be measured for a set of red, green, and blue images containing identified nuclei, 
measurements will be made between the following:
<ul>
<li>The blue and green, red and green, and red and blue images. </li>
<li>The nuclei in each of the above image pairs.</li>
</ul>
'''
# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
# 
# Developed by the Broad Institute
# Copyright 2003-2010
# 
# Please see the AUTHORS file for credits.
# 
# Website: http://www.cellprofiler.org

__version__="$Revision$"

import numpy as np
from scipy.linalg import lstsq
import scipy.ndimage as scind

import cellprofiler.cpmodule as cpm
import cellprofiler.objects as cpo
import cellprofiler.settings as cps
import cellprofiler.measurements as cpmeas
from cellprofiler.cpmath.cpmorphology import fixup_scipy_ndimage_result as fix

M_IMAGES = "Across entire image"
M_OBJECTS = "Within objects"
M_IMAGES_AND_OBJECTS = "Both"

'''Feature name format for the correlation measurement'''
F_CORRELATION_FORMAT = "Correlation_RWC_%s_%s"

'''Feature name format for the slope measurement'''
F_SLOPE_FORMAT = "Correlation_Slope_%s_%s"

class MeasureRWC(cpm.CPModule):

    module_name = 'MeasureRWC'
    category = 'Measurement'
    variable_revision_number = 1
    
    def create_settings(self):
        '''Create the initial settings for the module'''
        self.manual_threshold = cps.Float("Enter threshold:", .15)
        self.image_groups = []
        self.add_image(can_delete = False)
        self.spacer_1 = cps.Divider()
        self.add_image(can_delete = False)
        self.image_count = cps.HiddenCount(self.image_groups)
        
        self.add_image_button = cps.DoSomething("", 'Add another image', self.add_image)
        self.spacer_2 = cps.Divider()
        self.images_or_objects = cps.Choice('Select where to measure correlation',
                                            [M_IMAGES, M_OBJECTS, M_IMAGES_AND_OBJECTS], 
                                            doc = '''
                                            Do you want to measure the RWC over the whole image, 
                                            within objects, or both?
                                            Both methods measure RWC on a pixel by pixel basis.
                                            Selecting <i>Objects</i> will measure RWC only in those pixels previously
                                            identified as an object (you will be asked to specify which object).  Selecting 
                                            <i>Images</i> will measure RWC across all pixels in the images.
                                            <i>Images and objects</i> will calculate both measurements.''')
        
        self.object_groups = []
        self.add_object(can_delete = False)
        self.object_count = cps.HiddenCount(self.object_groups)
        
        self.spacer_2 = cps.Divider(line=True)
        
        self.add_object_button = cps.DoSomething("", 'Add another object', self.add_object)

    def add_image(self, can_delete = True):
        '''Add an image to the image_groups collection
        
        can_delete - set this to False to keep from showing the "remove"
                     button for images that must be present.
        '''
        group = cps.SettingsGroup()
        if can_delete:
            group.append("divider", cps.Divider(line=False))
        group.append("image_name", cps.ImageNameSubscriber('Select an image to measure','None',
                                                          doc = '''What is the name of an image to be measured?'''))
        if len(self.image_groups) == 0: # Insert space between 1st two images for aesthetics
            group.append("extra_divider", cps.Divider(line=False))
        
        if can_delete:
            group.append("remover", cps.RemoveSettingButton("","Remove this image", self.image_groups, group))
            
        self.image_groups.append(group)

    def add_object(self, can_delete = True):
        '''Add an object to the object_groups collection'''
        group = cps.SettingsGroup()
        if can_delete:
            group.append("divider", cps.Divider(line=False))
        group.append("object_name", cps.ObjectNameSubscriber('Select an object to measure','None',
                                                            doc = '''What is the name of objects to be measured?'''))
        if can_delete:
            group.append("remover", cps.RemoveSettingButton('', 'Remove this object', self.object_groups, group))
        self.object_groups.append(group)

    def settings(self):
        '''Return the settings to be saved in the pipeline'''
        result = [self.image_count, self.object_count]
        result += [image_group.image_name for image_group in self.image_groups]
        result += [self.images_or_objects]
        result += [object_group.object_name for object_group in self.object_groups]
        return result

    def prepare_settings(self, setting_values):
        '''Make sure there are the right number of image and object slots for the incoming settings'''
        image_count = int(setting_values[0])
        object_count = int(setting_values[1])
        if image_count < 2:
            raise ValueError("The MeasureCorrelate module must have at least two input images. %d found in pipeline file"%image_count)
        
        del self.image_groups[image_count:]
        while len(self.image_groups) < image_count:
            self.add_image()
        
        del self.object_groups[object_count:]
        while len(self.object_groups) < object_count:
            self.add_object()

    def visible_settings(self):
        result = []
        for image_group in self.image_groups:
            result += image_group.visible_settings()
        result += [self.add_image_button, self.spacer_2, self.images_or_objects, self.manual_threshold]
        if self.wants_objects():
            for object_group in self.object_groups:
                result += object_group.visible_settings()
            result += [self.add_object_button]
        return result

    def get_image_pairs(self):
        '''Yield all permutations of pairs of images to correlate
        
        Yields the pairs of images in a canonical order.
        '''
        for i in range(self.image_count.value-1):
            for j in range(i+1, self.image_count.value):
                yield (self.image_groups[i].image_name.value,
                       self.image_groups[j].image_name.value)

    def wants_images(self):
        '''True if the user wants to measure correlation on whole images'''
        return self.images_or_objects in (M_IMAGES, M_IMAGES_AND_OBJECTS)

    def wants_objects(self):
        '''True if the user wants to measure per-object correlations'''
        return self.images_or_objects in (M_OBJECTS, M_IMAGES_AND_OBJECTS)

    def run(self, workspace):
        '''Calculate measurements on an image set'''
        statistics = [["First image","Second image","Objects","Measurement","Value"]]
        for first_image_name, second_image_name in self.get_image_pairs():
            if self.wants_images():
                statistics += self.run_image_pair_images(workspace, 
                                                         first_image_name, 
                                                         second_image_name)
            if self.wants_objects():
                for object_name in [group.object_name.value for group in self.object_groups]:
                    statistics += self.run_image_pair_objects(workspace, 
                                                              first_image_name,
                                                              second_image_name, 
                                                              object_name)
        if not workspace.frame is None:
            figure = workspace.create_or_find_figure(title="MeasureRWC, image cycle #%d"%(
                workspace.measurements.image_set_number),subplots=(1,1))
            figure.subplot_table(0,0,statistics,(0.2,0.2,0.2,0.2,0.2))

    def run_image_pair_images(self, workspace, first_image_name, 
                              second_image_name):
        '''Calculate the correlation between the pixels of two images'''
        first_image = workspace.image_set.get_image(first_image_name,
                                                    must_be_grayscale=True)
        second_image = workspace.image_set.get_image(second_image_name,
                                                     must_be_grayscale=True)
        first_pixel_data = first_image.pixel_data
        first_mask = first_image.mask
        first_pixel_count = np.product(first_pixel_data.shape)
        second_pixel_data = second_image.pixel_data
        second_mask = second_image.mask
        second_pixel_count = np.product(second_pixel_data.shape)
        #
        # Crop the larger image similarly to the smaller one
        #
        if first_pixel_count < second_pixel_count:
            second_pixel_data = first_image.crop_image_similarly(second_pixel_data)
            second_mask = first_image.crop_image_similarly(second_mask)
        elif second_pixel_count < first_pixel_count:
            first_pixel_data = second_image.crop_image_similarly(first_pixel_data)
            first_mask = second_image.crop_image_similarly(first_mask)
        mask = (first_mask & second_mask & 
                (~ np.isnan(first_pixel_data)) &
                (~ np.isnan(second_pixel_data)))
        if np.any(mask):
          #
            # Perform the correlation, which returns:
            # [ [ii, ij],
            #   [ji, jj] ]
            #
            fi = first_pixel_data[mask]
            si = second_pixel_data[mask]
            #corr = np.corrcoef((fi,si))[1,0]

            
            
            
            # Do the ranking
            au,arank=np.unique(fi,return_inverse=True)
            bu,brank=np.unique(si,return_inverse=True)
            
            # Reverse ranking
            amax=np.max(arank)+1
            bmax=np.max(brank)+1
            arank = -(arank.astype(float)-amax)
            brank = -(brank.astype(float)-bmax)
            
            # Measure absolute difference in ranks
            d=np.absolute(arank-brank)
            
            # Get the maximal ranking
            rn=np.max(np.hstack((arank,brank)))
            
            # Calculate weights matrix
            w=(rn-d)/rn
            
            # Thresholding and RWC calculations
            t = self.manual_threshold.value
            #t=0.15
            ta=t*np.max(fi)
            tb=t*np.max(si)
            a1=np.array(fi, copy=True)
            b1=np.array(si, copy=True)
            a1[fi<=ta]=0
            asum=np.sum(a1)
            a1[si<=tb]=0
            rwc1=np.sum(a1.flatten()*w)/asum
            
            b1[si<=tb]=0
            bsum=np.sum(b1)
            b1[fi<=ta]=0
            rwc2=np.sum(b1.flatten()*w)/bsum

        else:
            rwc1 = np.NaN
            rwc2 = np.NaN
        #
        # Add the measurements
        #
        rwc1_measurement = F_CORRELATION_FORMAT%(first_image_name, 
                                                 second_image_name)
        rwc2_measurement = F_CORRELATION_FORMAT%(second_image_name,
                                                    first_image_name)
        workspace.measurements.add_image_measurement(rwc1_measurement, rwc1)
        workspace.measurements.add_image_measurement(rwc2_measurement, rwc2)
        return [[first_image_name, second_image_name, "-", "Correlation","%.2f"%rwc1],
                [second_image_name, first_image_name, "-", "Correlation","%.2f"%rwc2]]

    def run_image_pair_objects(self, workspace, first_image_name,
                               second_image_name, object_name):
        '''Calculate per-object correlations between intensities in two images'''
        first_image = workspace.image_set.get_image(first_image_name,
                                                    must_be_grayscale=True)
        second_image = workspace.image_set.get_image(second_image_name,
                                                     must_be_grayscale=True)
        objects = workspace.object_set.get_objects(object_name)
        #
        # Crop both images to the size of the labels matrix
        #
        labels = objects.segmented
        try:
            first_pixels  = objects.crop_image_similarly(first_image.pixel_data)
            first_mask    = objects.crop_image_similarly(first_image.mask)
        except ValueError:
            first_pixels, m1 = cpo.size_similarly(labels, first_image.pixel_data)
            first_mask, m1 = cpo.size_similarly(labels, first_image.mask)
            first_mask[~m1] = False
        try:
            second_pixels = objects.crop_image_similarly(second_image.pixel_data)
            second_mask   = objects.crop_image_similarly(second_image.mask)
        except ValueError:
            second_pixels, m1 = cpo.size_similarly(labels, second_image.pixel_data)
            second_mask, m1 = cpo.size_similarly(labels, second_image.mask)
            second_mask[~m1] = False
        mask   = ((labels > 0) & first_mask & second_mask)
        first_pixels = first_pixels[mask]
        second_pixels = second_pixels[mask]
        labels = labels[mask]
        if len(labels)==0:
            n_objects = 0
        else:
            n_objects = np.max(labels)
        if n_objects == 0:
            rwc1 = np.zeros((0,))
            rwc2 = np.zeros((0,))
        else:
            object_labels = np.unique(labels)
            rwc1 = np.zeros(np.shape(object_labels))
            rwc2 = np.zeros(np.shape(object_labels))
            for oindex in object_labels:
                fi = first_pixels[labels==oindex]
                si = second_pixels[labels==oindex]
                #corr = np.corrcoef((fi,si))[1,0]
                
                
                
                
                # Do the ranking
                au,arank=np.unique(fi,return_inverse=True)
                bu,brank=np.unique(si,return_inverse=True)
                
                # Reverse ranking
                amax=np.max(arank)+1
                bmax=np.max(brank)+1
                arank = -(arank.astype(float)-amax)
                brank = -(brank.astype(float)-bmax)
                
                # Measure absolute difference in ranks
                d=np.absolute(arank-brank)
                
                # Get the maximal ranking
                rn=np.max(np.hstack((arank,brank)))
                
                # Calculate weights matrix
                w=(rn-d)/rn
                
                # Thresholding and RWC calculations
                t = self.manual_threshold.value
                #t=0.15
                ta=t*np.max(fi)
                tb=t*np.max(si)
                a1=np.array(fi, copy=True)
                b1=np.array(si, copy=True)
                a1[fi<=ta]=0
                asum=np.sum(a1)
                a1[si<=tb]=0
                rwc1_temp=np.sum(a1.flatten()*w)/asum
                
                b1[si<=tb]=0
                bsum=np.sum(b1)
                b1[fi<=ta]=0
                rwc2_temp=np.sum(b1.flatten()*w)/bsum
                
                # And RWC values are...
                rwc1[oindex-1]= rwc1_temp
                rwc2[oindex-1]= rwc2_temp
                
        rwc1_measurement = ("Correlation_RWC_%s_%s" %
                       (first_image_name, second_image_name))
        rwc2_measurement = ("Correlation_RWC_%s_%s" %
                       (second_image_name, first_image_name))
        workspace.measurements.add_measurement(object_name, rwc1_measurement, rwc1)
        workspace.measurements.add_measurement(object_name, rwc2_measurement, rwc2)
        if n_objects == 0:
            return [[first_image_name, second_image_name, object_name,
                     "Mean correlation","-"],
                    [first_image_name, second_image_name, object_name,
                     "Median correlation","-"],
                    [first_image_name, second_image_name, object_name,
                     "Min correlation","-"],
                    [first_image_name, second_image_name, object_name,
                     "Max correlation","-"]]
        else:
            return [[first_image_name, second_image_name, object_name,
                     "Mean correlation","%.2f"%np.mean(rwc1)],
                     [first_image_name, second_image_name, object_name,
                     "Median correlation","%.2f"%np.median(rwc1)],
                     [first_image_name, second_image_name, object_name,
                     "Min correlation","%.2f"%np.min(rwc1)],
                     [first_image_name, second_image_name, object_name,
                     "Max correlation","%.2f"%np.max(rwc1)]]

    def get_measurement_columns(self, pipeline):
        '''Return column definitions for all measurements made by this module'''
        columns = []
        for first_image, second_image in self.get_image_pairs():
            if self.wants_images():
                columns += [(cpmeas.IMAGE,
                             F_CORRELATION_FORMAT%(first_image, second_image),
                             cpmeas.COLTYPE_FLOAT),
                            (cpmeas.IMAGE,
                             F_CORRELATION_FORMAT%(second_image, first_image),
                             cpmeas.COLTYPE_FLOAT)]
            if self.wants_objects():
                for i in range(self.object_count.value):
                    object_name = self.object_groups[i].object_name.value
                    columns += [(object_name,
                                 F_CORRELATION_FORMAT %
                                 (first_image, second_image),
                                 cpmeas.COLTYPE_FLOAT),
                                 (object_name,
                                 F_CORRELATION_FORMAT %
                                 (second_image, first_image),
                                 cpmeas.COLTYPE_FLOAT)]
        return columns

    def get_categories(self, pipeline, object_name):
        '''Return the categories supported by this module for the given object
        
        object_name - name of the measured object or cpmeas.IMAGE
        '''
        if ((object_name == cpmeas.IMAGE and self.wants_images()) or
            ((object_name != cpmeas.IMAGE) and self.wants_objects() and
             (object_name in [x.object_name.value for x in self.object_groups]))):
            return ["Correlation"]
        return [] 

    def get_measurements(self, pipeline, object_name, category):
        if self.get_categories(pipeline, object_name) == [category]:
            if object_name == cpmeas.IMAGE:
                return ["RWC"]
            else:
                return ["RWC"]
        return []

    def get_measurement_images(self, pipeline, object_name, category, 
                               measurement):
        '''Return the joined pairs of images measured'''
        if measurement in self.get_measurements(pipeline, object_name, category):
            result = []
            for x in self.get_image_pairs():
                result += "%s_%s"%(x[0],x[1])
                result += "%s_%s"%(x[1],x[0])
            #return ["%s_%s"%x for x in self.get_image_pairs()]
            return result
        return []

