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
import numpy as np
from glob2 import glob
import os
from joblib import Parallel, delayed
import pandas as pd
import pdal
import json
from osgeo import gdal

def pdal_features(incld, outcld, k=8, nt=8, writer='las', optim=True,
                  features='all'):
    
    """ 
    Calculate pdal-based point features and write to a new cloud MUST be las
    at present
       
    Parameters 
    ----------- 
    
    incld: string
              the input point cloud
        
    outcld: string
               the output point cloud if None then write to incld
               
    k: int
            the no of neighbours (8)
            
    nt: int
            no of threads
    
    writer: file type to write
            e.g. ply, las
    
    optim: bool
            enable computation of features using precomputed optimal neighborhoods
    
    features: string
                features to compute  default is 'all'
    """ 
    
    
    js = [incld,
        {
            "type":"filters.optimalneighborhood"
        },
        {
            "type":"filters.covariancefeatures",
            "knn":k,
            "threads": nt,
            "optimized": optim,
            "feature_set": features
        },
        {
            "type":"writers."+writer,
            "filename": outcld
        }
        ]
    # RuntimeError: writers.ply: Can't write dimension of type 'uint64_t'
    if writer == 'ply':
        js[3]["precision"] = "6f"
        
    # for las somewhere to put these fts
    if writer == 'las':
        js[3]["extra_dims"] = "all"
        js[3]["forward"] = "all"
        js[3]["minor_version"] = 4
    
    pipeline = pdal.Pipeline(json.dumps(js))
    count = pipeline.execute()


def cgal_features(incld, outcld=None, k=5, rgb=True, parallel=True):
    
    """ 
    Calculate CGAL-based point cloud features and write to file.
    
    Files will be hefty! 
       
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
            
    parallel: bool
            if true, process multi thread
    
    tofile: bool
            whether to write the attributes to file or return in memory

    """ 

    print("Reading pointcloud")
    points = Point_set_3(incld)
 
    print("Computing features")
    
    features = Feature_set()
    generator = Point_set_feature_generator(points, k)
    
    if parallel is True:
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
    if parallel is True:
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
        


def cgal_features_tile(folder, k=5, rgb=True,  nt=None):
    
    """ 
    Calculate CGAL-based point cloud features for a folder containing ply files
    Feature attributes will be written to the input ply files.
    
    Files will be hefty!
       
    Parameters 
    ----------- 
    
    folder: string
              the input folder containing .ply tiles
        
    k: int
            he no of scales at which to calculate features

    rgb: bool
            whether to include RGB-based features
    
    nt: int
            number of threads to use in processing

    """ 
    
    plylist = glob(os.path.join(folder, '*.ply'))
    plylist.sort()
    
    if nt == None:
        nt = len(plylist)
        
    Parallel(n_jobs=nt, verbose=2)(delayed(cgal_features)(p,  k=k, rgb=rgb, 
             parallel=False) for p in plylist)



def cgal_features_mem(incld,  k=5, rgb=True, parallel=True):
    
    """ 
    Calculate CGAL-based point cloud features and return as a pandas df. 
    Saves on disk space but many of these will take longer.
       
    Parameters 
    ----------- 
    
    incld: string
              the input point cloud
        
    k: int
            he no of scales at which to calculate features

    rgb: bool
            whether to include RGB-based features
            
    parallel: bool
            if true, process multi thread
    
    tofile: bool
            whether to write the attributes to file or return in memory

    """ 

    print("Reading pointcloud")
    points = Point_set_3(incld)
 
    print("Computing features")
    
    features = Feature_set()
    generator = Point_set_feature_generator(points, k)
    
    #TODO  Not convinced this is actually running more than 1 thread
    if parallel is True:
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
    if parallel is True:
        features.end_parallel_additions()
    
    print("Features calculated")
       
    names = _get_featnames(features)
    
    # could return this or the feat names....
    featarray = np.zeros(shape=(points.size(),len(names))) 
    
    # for ref in case ressurrected
    #df = pd.DataFrame(columns=[names])
    # if we are pulling from a shared mem object can we do in para
    # oh yeah cant pickle a swig object.....then inserting a list into a df is
    # much slower
    #cnt = np.arange(0, points.size())
    #bigList = Parallel(n_jobs=nt, verbose=2)(
    #delayed(_cgalfeat)(cnt, features, n, idx) for idx, n in enumerate(names))

    # ~ 8 minutes for 6.4 million points
    for ftr, r in enumerate(tqdm(names)):
        # loops in loops, this is not ideal - 
        # TODO need c++ func to output np array
        featarray[:,ftr] = [features.get(ftr).value(i) for i,
                 p in enumerate(points.indices())]
    
    df = pd.DataFrame(data=featarray, columns=[names])
    
    return df


def std_features(incld, outcld=None, k=[50,100,200],
                 props=['anisotropy', "curvature", "eigenentropy", "eigen_sum",
                         "linearity","omnivariance", "planarity", "sphericity"],
                        nrm_props=None, tofile=True):
    
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
    
    tofile: bool
            if true write to pointcloud if false return a df of the features

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
        

    if tofile !=True:
        # create strings with k added in brackets
        # TODO replace with something more elegant
        df = pcd.points
        eigens = ['e1', 'e2', 'e3'] +pProps
        keep = []
        for i in k:
            # pyntcloud seems to add ('1') to scales 
            strn = "("+str(i+1)+")"
            keep.append([e+strn for e in eigens])
        keep = keep[0] + keep[1]  
        
        # use the set function to get the cols we wish to remove
        colsorig = list(df.columns)
        dumped = list(set(colsorig) - set(keep))
        
        return df.drop(columns=dumped)
    else:

        if outcld == None:
            pcd.to_file(incld)
        else:
            pcd.to_file(outcld)
            



def _get_featnames(features):
    """
    get the feature names
    """
    numoffeat = features.size()
    featcols = []
    
    for idx in range(min(numoffeat, features.size())):
        featcols.append(features.get(idx).name())
    return featcols

def _feat_row(features, names, idx):
    
    """
    get the row of values
    """   
    cnt = np.arange(features.size()).tolist()
    
    oot = [features.get(c).value(idx) for c in cnt]

# This doesn't result in an aggregate speed up as the entries must still 
# be written to disk, but is here a a reference nontheless
def _featgen(points):
    """
    write the features to list
    """
    
    ar = np.arange(points.size()).tolist()
    
    feat_out = [_feat_row(features, names, i) for i in tqdm(ar)] 


def _label_transfer(incld, labelcloud, field='training'):
    
    """ get labels from one cloud and add to another if you
    forget to close down cgal correctly or something
    
    """
    
    pcd = PyntCloud.from_file(incld)
    
    pcd2 = PyntCloud.from_file(labelcloud)
    
    pcd.points['training'] = pcd2.points['training']
    
    pcd.to_file(incld)
    

# not used...
def _cgalfeat(cnt, features, name, ftr):
    
    flist = [features.get(ftr).value(i) for i in cnt]
    
    return flist  #np.asarray(flist)
    
    
def write_vrt(infiles, outfile):
    
    """
    Parameters
    ----------
    
    infiles: list of strings
                the input files
    
    outfile: string
                the output .vrt

    """
    
    
    virtpath = outfile
    outvirt = gdal.BuildVRT(virtpath, infiles)
    outvirt.FlushCache()
    outvirt=None   
    

    
