'''<b>MeasurementTAS</b> - Measure Threshold adjacency statistics
<hr>
This module measures the Threshold Adjacency Statistics of an image, object or both.

Please cite the following paper:
Hamilton NA, Pantelic RS, Hanson K, Teasdale RD.: Fast automated cell phenotype image classification.
BMC Bioinformatics 2007, 8:110.


'''
#################################
#
# Imports from useful Python libraries
#
#################################

import numpy as np
import scipy.ndimage as scind

#################################
#
# Imports from CellProfiler
#
# The package aliases are the standard ones we use
# throughout the code.
#
##################################

import cellprofiler.cpimage as cpi
import cellprofiler.cpmodule as cpm
import cellprofiler.measurements as cpmeas
import cellprofiler.objects as cpo
import cellprofiler.settings as cps

#from cellprofiler.cpmath.zernike import construct_zernike_polynomials
#from cellprofiler.cpmath.cpmorphology import minimum_enclosing_circle
from cellprofiler.cpmath.cpmorphology import fixup_scipy_ndimage_result as fix

##################################
#
# Constants
#
# I put constants that are used more than once here.
#
# The Zernike list here is a list of the N & M Zernike
# numbers that we use below in our measurements. I made it
# a constant because the same list will be used to make
# the measurements and to report the measurements that can
# be made.
#
###################################
#ZERNIKE_LIST = ((0, 0), (1, 1), (2, 0), (2, 2), (3, 1), (3, 3), 
#                (4, 0), (4, 2), (4, 4), (5, 1), (5, 3), (5, 5))
TAS_LIST = ((('Low',-0.118),('High',0.118)),(('Low',-0.118),('Inf',1)),(('Mean',0),('Inf',0.118)))
TAS_WEIGHTS = range(9)
'''This is the measurement template category'''
C_MEASUREMENT_TEMPLATE = "MT"

###################################
#
# The module class
#
# Your module should "inherit" from cellprofiler.cpmodule.CPModule.
# This means that your module will use the methods from CPModule unless
# you re-implement them. You can let CPModule do most of the work and
# implement only what you need.
#
###################################

class MeasurementTAS(cpm.CPModule):
    ###############################################
    #
    # The module starts by declaring the name that's used for display,
    # the category under which it is stored and the variable revision
    # number which can be used to provide backwards compatibility if
    # you add user-interface functionality later.
    #
    ###############################################
    module_name = "MeasurementTAS"
    category = "Measurement"
    variable_revision_number = 1
    
    ###############################################
    #
    # create_settings is where you declare the user interface elements
    # (the "settings") which the user will use to customize your module.
    #
    # You can look at other modules and in cellprofiler.settings for
    # settings you can use.
    #
    ################################################
    
    def create_settings(self):
        #
        # The ImageNameSubscriber "subscribes" to all ImageNameProviders in 
        # prior modules. Modules before yours will put images into CellProfiler.
        # The ImageSubscriber gives your user a list of these images
        # which can then be used as inputs in your module.
        #
        self.manual_threshold = cps.Float("Enter threshold:", .118)
        self.input_image_name = cps.ImageNameSubscriber(
            # The text to the left of the edit box
            "Input image name:",
            # HTML help that gets displayed when the user presses the
            # help button to the right of the edit box
            doc = """This is the image that the module operates on. You can
            choose any image that is made available by a prior module.
            """)
        
        #
        # The ObjectNameSubscriber is similar - it will ask the user
        # which object to pick from the list of objects provided by
        # upstream modules.
        #
        self.input_object_name = cps.ObjectNameSubscriber(
            "Input object name",
            doc = """These are the objects that the module operates on.""")
        
    #
    # The "settings" method tells CellProfiler about the settings you
    # have in your module. CellProfiler uses the list for saving
    # and restoring values for your module when it saves or loads a
    # pipeline file.
    #
    # This module does not have a "visible_settings" method. CellProfiler
    # will use "settings" to make the list of user-interface elements
    # that let the user configure the module. See imagetemplate.py for
    # a template for visible_settings that you can cut and paste here.
    #
    def settings(self):
        return [self.input_image_name, self.input_object_name]
    
    #
    # CellProfiler calls "run" on each image set in your pipeline.
    # This is where you do the real work.
    #
    def run(self, workspace):
        #
        # Get the measurements object - we put the measurements we
        # make in here
        #
        meas = workspace.measurements
        #assert isinstance(meas, cpmeas.Measurements)
        #
        # We record some statistics which we will display later.
        # We format them so that Matplotlib can display them in a table.
        # The first row is a header that tells what the fields are.
        #
        #statistics = [ [ "Feature", "Mean", "Median", "SD"] ]
        #
        # Put the statistics in the workspace display data so we
        # can get at them when we display
        #
        #workspace.display_data.statistics = statistics
        #
        # Get the input image and object. You need to get the .value
        # because otherwise you'll get the setting object instead of
        # the string name.
        #
        input_image_name = self.input_image_name.value
        input_object_name = self.input_object_name.value
        ################################################################
        #
        # GETTING AN IMAGE FROM THE IMAGE SET
        #
        # Get the image set. The image set has all of the images in it.
        # The assert statement makes sure that it really is an image set,
        # but, more importantly, it lets my editor do context-sensitive
        # completion for the image set.
        #
        image_set = workspace.image_set
        #assert isinstance(image_set, cpi.ImageSet)
        #
        # Get the input image object. We want a grayscale image here.
        # The image set will convert a color image to a grayscale one
        # and warn the user.
        #
        input_image = image_set.get_image(input_image_name,
                                          must_be_grayscale = True)
        #
        # Get the pixels - these are a 2-d Numpy array.
        #
        pixels = input_image.pixel_data
        #
        ###############################################################
        #
        # GETTING THE LABELS MATRIX FROM THE OBJECT SET
        #
        # The object set has all of the objects in it.
        #
        object_set = workspace.object_set
        #assert isinstance(object_set, cpo.ObjectSet)
        #
        # Get objects from the object set. The most useful array in
        # the objects is "objects.segmented" which is a labels matrix
        # in which each pixel has an integer value.
        #
        # The value, "0", is reserved for "background" - a pixel with
        # a zero value is not in an object. Each object has an object
        # number, starting at "1" and each pixel in the object is
        # labeled with that number.
        #
        # The other useful array is "objects.small_removed_segmented" which
        # is another labels matrix. There are objects that touch the edge of
        # the image and get filtered out and there are large objects that
        # get filtered out. Modules like "IdentifySecondaryObjects" may
        # want to associate pixels near the objects in the labels matrix to
        # those objects - the large and touching objects should compete with
        # the real ones, so you should use "objects.small_removed_segmented"
        # for those cases.
        #
        objects = object_set.get_objects(input_object_name)
        labels = objects.segmented
        #
        ###############################################################
        #
        # The module computes a measurement based on the image intensity
        # inside an object times a Zernike polynomial inscribed in the
        # minimum enclosing circle around the object. The details are
        # in the "measure_zernike" function. We call into the function with
        # an N and M which describe the polynomial.
        #
        for n, m in TAS_LIST:
            # Compute the TAS for each object, returned in an array
            tas = self.measure_TAS(pixels, labels, n[1], m[1])
            # Add a measurement for this kind of object
            #for i, z in enumerate(tas):
                # Get the name of the measurement feature for this zernike
           #     feature = self.get_measurement_name(n[0], m[0], i)
            #    meas.add_measurement(input_object_name, feature, z)
            #
            
            for i, t in enumerate(tas):
                feature = self.get_measurement_name(n[0], m[0], i)
                meas.add_measurement(input_object_name, feature, t)
            
            
            # Record the statistics. 
            #
    ################################
    # 
    # DISPLAY
    #
    # We define is_interactive to be False to tell CellProfiler
    # that it should execute "run" in a background thread and then
    # execute "display" in a foreground thread.
    #
    def is_interactive(self):
        return False
    
    #def display(self, workspace):
    ##    statistics = workspace.display_data.statistics
    #    figure = workspace.create_or_find_figure(subplots=(1,1,))
    #    figure.subplot_table(0,0, statistics, ratio = (.25, .25, .25, .25))
    
    ################################
    #
    # measure_zernike makes one Zernike measurement on each object
    # 
    def measure_TAS(self, pixels, labels, n, m):
        # I'll put some documentation in here to explain what it does.
        # If someone ever wants to call it, their editor might display
        # the documentation.
        '''Measure the intensity of the image with Zernike (N, M)
        
        pixels - the intensity image to be measured
        labels - the labels matrix that labels each object with an integer
        n, m - the Zernike coefficients.
        
        See http://en.wikipedia.org/wiki/Zernike_polynomials for an
        explanation of the Zernike polynomials
        '''
        #
        # The strategy here is to operate on the whole array instead
        # of operating on one object at a time. The most important thing
        # is to avoid having to run the Python interpreter once per pixel
        # in the image and the second most important is to avoid running
        # it per object in case there are hundreds of objects.
        #
        # We play lots of indexing tricks here to operate on the whole image.
        # I'll try to explain some - hopefully, you can reuse.
        #
        # You could move the calculation of the minimum enclosing circle
        # outside of this function. The function gets called more than
        # 10 times, so the same calculation is performed 10 times.
        # It would make the code a little more confusing, so I'm leaving
        # it as-is.
        ###########################################
        #
        # The minimum enclosing circle (MEC) is the smallest circle that
        # will fit around the object. We get the centers and radii of
        # all of the objects at once. You'll see how that lets us
        # compute the X and Y position of each pixel in a label all at
        # one go.
        #
        # First, get an array that lists the whole range of indexes in
        # the labels matrix.
        #
        
        if len(labels)==0:
            n_objects = 0
        else:
            n_objects = np.max(labels)
            
        if n_objects==0:
            result = np.zeros((0,))
        else:
        
            indexes = np.arange(1, np.max(labels)+1,dtype=np.int32)
            
            # Calculate mean of pixels above int 30
            mean=np.mean(pixels[pixels>0.118])
            
            # Set ranges
            rangelow = mean + n
            rangehigh = mean + m
            
            # Threshold image, create mask
            mask=np.logical_and(pixels<rangehigh,pixels>rangelow)
            thresholded_image=np.zeros(np.shape(pixels))
            thresholded_image[mask]=1
            
            # Apply convolution to get the sum of sorrounding pixels
            # First define a weight array
            w=np.array([[1,1,1],[1,0,1],[1,1,1]])
            # Create a new array of sums
            sums=scind.convolve(thresholded_image,w,mode='constant')
            # remove 0 pixels from sums
            sums[~mask]=9
            
            # Get the histogram
            result = fix(scind.histogram(sums.astype(int),0,9,10,labels=labels, index=indexes))
            #print np.shape(result)
            result = np.vstack(result).T.astype(np.float64)
            #result = np.random.rand(2,9).T
            result = result/np.sum(result,axis=0)
            #result = result[0:9]
            #result = (result/np.sum(result)).astype(np.float64)
            #
            # And we're done! Did you like it? Did you get it?
            #
        return result[0:9]
    
    #######################################
    #
    # Here, we go about naming the measurements.
    #
    # Measurement names have parts to them, traditionally separated
    # by underbars. There's always a category and a feature name
    # and sometimes there are modifiers such as the image that
    # was measured or the scale at which it was measured.
    #
    # We have functions that build the names so that we can
    # use the same functions in different places.
    #
    def get_feature_name(self, n, m, i=None):
        '''Return a measurement feature name for the given Zernike'''
        #
        # Something nice and simple for a name... IntensityN4M2 for instance
        #
        return "TextureTAS%s%s%s" % (n, m, i)
    
    def get_measurement_name(self, n, m, i=None):
        '''Return the whole measurement name'''
        input_image_name = self.input_image_name.value
        return '_'.join([C_MEASUREMENT_TEMPLATE, 
                         self.get_feature_name(n,m,i),
                         input_image_name])
    #
    # We have to tell CellProfiler about the measurements we produce.
    # There are two parts: one that is for database-type modules and one
    # that is for the UI. The first part gives a comprehensive list
    # of measurement columns produced. The second is more informal and
    # tells CellProfiler how to categorize its measurements.
    #
    #
    # get_measurement_columns gets the measurements for use in the database
    # or in a spreadsheet. Some modules need the pipeline because they
    # might make measurements of measurements and need those names.
    #
    def get_measurement_columns(self, pipeline):
        #
        # We use a list comprehension here.
        # See http://docs.python.org/tutorial/datastructures.html#list-comprehensions
        # for how this works.
        #
        # The first thing in the list is the object being measured. If it's
        # the whole image, use cpmeas.IMAGE as the name.
        #
        # The second thing is the measurement name.
        #
        # The third thing is the column type. See the COLTYPE constants
        # in measurements.py for what you can use
        #
        input_object_name = self.input_object_name.value
        return [ (input_object_name,
                  self.get_measurement_name(n[0], m[0], i),
                  cpmeas.COLTYPE_FLOAT)
                 #for n, m in TAS_LIST for i in TAS_WEIGHTS]
                 for n, m in TAS_LIST for i in TAS_WEIGHTS]
    
    #
    # get_categories returns a list of the measurement categories produced
    # by this module. It takes an object name - only return categories
    # if the name matches.
    #
    def get_categories(self, pipeline, object_name):
        if object_name == self.input_object_name:
            return [ C_MEASUREMENT_TEMPLATE ]
        else:
            # Don't forget to return SOMETHING! I do this all the time
            # and CP mysteriously bombs when you use ImageMath
            return []

    #
    # Return the feature names if the object_name and category match
    #
    def get_measurements(self, pipeline, object_name, category):
        if (object_name == self.input_object_name and
            category == C_MEASUREMENT_TEMPLATE):
            #
            # Use another list comprehension. See docs in get_measurement_columns.
            return [ self.get_feature_name(n[0], m[0], i)
                     for n, m in TAS_LIST for i in TAS_WEIGHTS]
        else:
            return []
        
    #
    # This module makes per-image measurements. That means we need
    # get_measurement_images to distinguish measurements made on two
    # different images by this module
    #
    def get_measurement_images(self, pipeline, object_name, category, measurement):
        #
        # This might seem wasteful, but UI code can be slow. Just see
        # if the measurement is in the list returned by get_measurements
        #
        if measurement in self.get_measurements(
            pipeline, object_name, category):
            return [ self.input_image_name.value]
        else:
            return []
