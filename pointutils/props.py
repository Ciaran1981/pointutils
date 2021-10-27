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
            
def grid_cloud(incld, outfile, attribute="label", reader="readers.ply",
               writer="writers.gdal", spref="EPSG:21818", dtype="uint16_t",
               outtype='mean', resolution=0.1):
    
    """
    Grid a pointcloud attribute using pdal
    
    Parameters
    ----------
    
    incld: string
            input cloud
    
    outfile: string
            output cloud
    
    attribute: string
            the pointcloud attribute/dimension to rasterize
            e.g. label, classification etc
    
    reader: string
            the pdal reader type (see pdal readers)
    
    writer: string
            the pdal reader type (see pdal writers)
    
    spref: string
            spatial ref in ESPG format
    
    dtype: string
            dtype in pdal format (see pdal)
        
    outtype: string
            mean, min or max
            
    resolution: float
            in the unit required

    """
    
    #json from args
    js = {
          "pipeline":[
            {
                "type": reader,
                "filename":incld,
        	"spatialreference":spref
            },
            {
              "type": writer, 
              "filename": outfile,
              "dimension": attribute,
              "data_type": dtype,
              "output_type": outtype,
              "resolution": resolution
            }
          ]
        }  
    
    
    pipeline = pdal.Pipeline(json.dumps(js))
    count = pipeline.execute()
    #log = pipeline.log
    
def grid_cloud_tile(folder, attribute="label", reader="readers.ply", 
                    writer="writers.gdal", spref="EPSG:21818", 
                    dtype="uint16_t", outtype='mean', resolution=0.1, nt=-1):
    
    """
    Grid pointclouds attribute using pdal
    
    Parameters
    ----------
    
    folder: string
            input folder containing plys
    
    attribute: string
            the pointcloud attribute/dimension to rasterize
            e.g. label, classification etc
    
    reader: string
            the pdal reader type (see pdal readers)
    
    writer: string
            the pdal reader type (see pdal writers)
    
    spref: string
            spatial ref in ESPG format
    
    dtype: string
            dtype in pdal format (see pdal)
        
    outtype: string
            mean, min or max
            
    resolution: float
            in the unit required

    """
    
    
    plylist = glob(os.path.join(folder, '*.ply'))
    plylist.sort()

    outlist = [ply[:-3]+'tif' for ply in plylist]
        
        
    Parallel(n_jobs=nt, verbose=2)(delayed(grid_cloud)(ply, out, attribute,
             reader, writer, spref, dtype,
             outtype, resolution) for ply, out in zip(plylist, outlist))



def pdal_ground(incld, smrf=None, scalar=1.25, slope=0.15, threshold=0.5, 
               window=18, clsrange="[1:2]", outcld=None):
    """
    Parameters
    ----------
    
    incld: string
            input cloud
    
    outcld: string
            output cloud (if none), results written to input
    
    smrf: int 
                From the Pingel (2013) paper tests - helpful perhaps but no
                guarantees of 'good' results!
                Choice of: 
                1 = Mixed vegetation and buildings on hillside,
                2 = Mixed vegetation and buildings,
                3 = Road with bridge,
                4 = Bridge and irregular ground surface,
                5 = Large, irregularly shaped buildings,
                6 = Steep slopes with vegetation,
                7 = Complex building,
                8 = Large gaps in data, irregularly shaped buildings,
                9 = Trains in railway yard,
                10 = Data gaps, vegetation on moderate slopes,
                11 = Steep, terraced slopes1
                12 = Steep, terraced slopes2
                13 = Dense ground cover
                14 = Large gap in data
                15 = Underpass
                    
    scalar: float
            scaling factor (def 1.25)
    
    slope: float
            slope thresh (def 0.15)
    
    threshold: float
            elevation threshold (def 0.5)

    window: int
            max window size  (def 18)
    """
    
    # the test results from the paper

    if outcld == None:
        outcld = incld
    if smrf_params != None:
        sms = smrf_params()
        row = sms.iloc[smrf]
        print('Parametrs are:\n', row[2:6])
        params = row[2:6].to_dict()
        params["type"] ="filters.smrf"

    else:
         params = {
            "type":"filters.smrf",
            "scalar":scalar,
            "slope":slope,
            "threshold":threshold,
            "window":window
        }
        
    js = [
        incld,
        params,
        {
            "type":"filters.range",
            "limits":"Classification"+clsrange
        },
        outcld
    ]
    
    pipeline = pdal.Pipeline(json.dumps(js))
    pipeline.execute()


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
    
# TODO this needs replaced
def smrf_params():
    
    """
    Parameters
    ----------
    None
    
    Returns
    -------
    
    Dataframe of the parameters from the Pingle2013 paper
    
    """
    
    
    params = {'Sample': {0: '1-1',
              1: '1-2',
              2: '2-1',
              3: '2-2',
              4: '2-3',
              5: '2-4',
              6: '3-1',
              7: '4-1',
              8: '4-2',
              9: '5-1',
              10: '5-2',
              11: '5-3',
              12: '5-4',
              13: '6-1',
              14: '7-1'},
             'Features': {0: 'Mixed vegetation and buildings on hillside',
              1: 'Mixed vegetation and buildings',
              2: 'Road with bridge',
              3: 'Bridge and irregular ground surface',
              4: 'Large, irregularly shaped buildings',
              5: 'Steep slopes with vegetation',
              6: 'Complex building',
              7: 'Large gaps in data, irregularly shaped buildings',
              8: 'Trains in railway yard',
              9: 'Data gaps, vegetation on moderate slopes',
              10: 'Steep, terraced slopes',
              11: 'Steep, terraced slopes',
              12: 'Dense ground cover',
              13: 'Large gap in data',
              14: 'Underpass'},
             'slope': {0: 0.2,
              1: 0.18,
              2: 0.12,
              3: 0.16,
              4: 0.27,
              5: 0.16,
              6: 0.08,
              7: 0.22,
              8: 0.06,
              9: 0.05,
              10: 0.13,
              11: 0.45,
              12: 0.05,
              13: 0.28,
              14: 0.13},
             'window': {0: 16,
              1: 12,
              2: 20,
              3: 18,
              4: 13,
              5: 8,
              6: 15,
              7: 16,
              8: 49,
              9: 17,
              10: 13,
              11: 3,
              12: 11,
              13: 5,
              14: 15},
             'threshold': {0: 0.45,
              1: 0.3,
              2: 0.6,
              3: 0.35,
              4: 0.5,
              5: 0.2,
              6: 0.25,
              7: 1.1,
              8: 1.05,
              9: 0.35,
              10: 0.25,
              11: 0.1,
              12: 0.15,
              13: 0.5,
              14: 0.75},
             'scalar': {0: 1.2,
              1: 0.95,
              2: 0.0,
              3: 1.3,
              4: 0.9,
              5: 2.05,
              6: 1.5,
              7: 0.0,
              8: 0.0,
              9: 0.9,
              10: 2.2,
              11: 3.8,
              12: 2.3,
              13: 1.45,
              14: 0.0}}
    df = pd.DataFrame(params)
    # error pdal doesn't like int64 for serialise...
    df['window'] = df['window'].astype(float)
    return df
    
