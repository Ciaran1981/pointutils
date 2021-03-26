"""
author: Ciaran Robb

props module

Description
-----------

Point cloud properties and manipulation in python

"""

from CGAL.CGAL_Kernel import Point_3
from CGAL.CGAL_Kernel import Vector_3
from CGAL.CGAL_Point_set_3 import Point_set_3
from CGAL.CGAL_Classification import *
from tqdm import tqdm
from pyntcloud import PyntCloud

# The following is very ugly - replace it.
def _get_featnames(features):
    """
    get the feature names
    """
    numoffeat = features.size()
    featcols = []
    
    for idx in range(min(numoffeat, features.size())):
        featcols.append(features.get(idx).name())
    return featcols

def cgal_features(incld, outcld=None, k=5, rgb=True):
    
    """ 
    Calculate CGAL-based point cloud features and write to file.
       
    Parameters 
    ----------- 
    
    incld: string
              the input point cloud
        
    outcld: string
               the output point cloud if None then write to incld
    k: int
            he no of scales at which to calculate features

    rgb: bool
            whether to include RGB-based features

    """ 

    print("Reading pointcloud")
    points = Point_set_3(incld)
 
    print("Computing features")
    
    features = Feature_set()
    generator = Point_set_feature_generator(points, k)
    
    features.begin_parallel_additions()
    generator.generate_point_based_features(features)
    if points.has_normal_map():
        generator.generate_normal_based_features(features, points.normal_map())
    
    if rgb is True:
        if points.has_int_map("red") and points.has_int_map("green") and points.has_int_map("blue"):
            generator.generate_color_based_features(features,
                                                    points.int_map("red"),
                                                    points.int_map("green"),
                                                    points.int_map("blue"))
    features.end_parallel_additions()
    
    print("Features calculated")
       
    names = _get_featnames(features)
    
    
    if rgb is True:
        
        rgbList = [points.add_float_map(n) for n in names[-3:]]
        
        for ftr, r in enumerate(tqdm(rgbList)):
            ftr+=50 # not ideal
            # list comp is no quicker...
#            _ = [r.set(p, features.get(ftr).value(i)) for i,
#                 p in enumerate(points.indices())]
            for i, p in enumerate(points.indices()):
                r.set(p, features.get(ftr).value(i))     
        # scrub the hsv
        del names[-3:]

    # to go in the loop below
    attribList = [points.add_float_map(n) for n in names]

    # This is of course unacceptably slow.
    # TODO C++ function and wrap
    for ft, a in enumerate(tqdm(attribList)):  
        
        # is list comp is quicker - NOT REALLY
        # ~11-12 minutes for 6.4million points w/ lcomp
        #_ = [a.set(p, features.get(ft).value(i)) for i, p in enumerate(points.indices())]
        
        # ~10 minutes for 6.4 million points with standard loop
        for i, p in enumerate(points.indices()):
            a.set(p, features.get(ft).value(i))

    
    if outcld == None:
        outcld = incld
    points.write(outcld)
    
    
def std_features(incld, outcld=None, k=[50,100,200],
                 props=['anisotropy', "curvature", "eigenentropy", "eigen_sum",
                         "linearity","omnivariance", "planarity", "sphericity"],
                        nrm_props=None):
    
    """ 
    Calculate point cloud features and write to file, over a range of k-scales
    
    'anisotropy', "curvature", "eigenentropy", "eigen_sum",
    "linearity","omnivariance", "planarity", "sphericity"
    
    Parameters 
    ----------- 
    
    incld: string
              the input point cloud
        
    outcld: string
               the output point cloud
    k: list
              the no of neighbors to use when calculating the props
              multiple is more effective
    props: list
            the properties you wish to include

    nrm_props: list
            properties based on normals if the exist (this will fail if they don't)
            e.g. ["inclination_radians",  "orientation_radians"]

    """  
    
    pcd = PyntCloud.from_file(incld)

    pProps = props 
    
    # For ref
    # pProps =['anisotropy', "curvature", "eigenentropy", "eigen_sum", "linearity",
    #          "omnivariance", "planarity", "sphericity"]#, "inclination_deg",
    #          "inclination_rad", "orientation_deg", "orientation_rad"]
    #, "HueSaturationValue",#"RelativeLuminance"," RGBIntensity"]
    
    
    # iterate through neighborhood sizes to get a multiscale output
    for i in k:
        k_neighbors = pcd.get_neighbors(k=i)
        eigenvalues = pcd.add_scalar_field("eigen_values", k_neighbors=k_neighbors)
        [pcd.add_scalar_field(p, ev=eigenvalues) for p in pProps]
    
    if nrm_props != None:
        [pcd.add_scalar_field(n) for n in nrm_props]
        
        
    
    if outcld == None:
        pcd.to_file(incld)
    else:
        pcd.to_file(outcld)

def _label_transfer(incld, labelcloud, field='training'):
    
    """ get labels from one cloud and add to another if you
    forget to close down cgal correctly or something
    
    """
    
    pcd = PyntCloud.from_file(incld)
    
    pcd2 = PyntCloud.from_file(labelcloud)
    
    pcd.points['training'] = pcd2.points['training']
    
    pcd.to_file(incld)
    
    
    
    
    
    
    
    
    
