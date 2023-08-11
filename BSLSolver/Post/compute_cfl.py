# ///////////////////////////////////////////////////////////////
#  Courant Number Calculation
#  Author: Mehdi Najafi, mnajafi@sharif.edu
#  Date: 2008-02-11
#
#  This library is not intended for distributions in any form and
#  distribution of it is not allowed in any form.
# ///////////////////////////////////////////////////////////////

__author__ = "Mehdi Najafi <mnajafi@sharif.edu>" #altered Anna Haley 2023
__date__ = "2008-02-11" #2023
__copyright__ = "Copyright (C) 2008 " + __author__
__license__  = "Private; to be obtained directly from the author."

import sys, os
# os.environ['OPENBLAS_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'

import h5py, glob
import numpy
import gc
import job_utils
import multiprocessing

################################################################
# get the element dimensions
#  dt*(u/dx+v/dy+w/dz)<=1
#  CFL = dt * (u/min(dx) + v/min(dy) + w/min(dz))
################################################################
# get the element dimensions
def get_mesh_deltas(mesh_h5_filename):
    mesh = h5py.File(mesh_h5_filename, 'r')
    cells = numpy.asarray(mesh['/Mesh/topology'])

    # look for previously generated one and load it
    if 'elem_dx' in mesh['/Mesh'].keys():
        return numpy.asarray(mesh['/Mesh/elem_dx']), cells

    points = numpy.asarray(mesh['/Mesh/coordinates'])
    number_of_cells = cells.shape[0]
    mesh_dx = numpy.zeros((number_of_cells,4), dtype=numpy.float64)
    for k in range(number_of_cells):
        dx = [1E20,1E20,1E20,0]
        point_ids = cells[k]
        for i in range(point_ids.shape[0]):
            for j in range(i+1,point_ids.shape[0]):
                ddx = numpy.fabs(points[point_ids[i]][0] - points[point_ids[j]][0])
                ddy = numpy.fabs(points[point_ids[i]][1] - points[point_ids[j]][1])
                ddz = numpy.fabs(points[point_ids[i]][2] - points[point_ids[j]][2])
                if ddx > 1E-9: dx[0] = min(dx[0], ddx)
                if ddy > 1E-9: dx[1] = min(dx[1], ddy)
                if ddz > 1E-9: dx[2] = min(dx[2], ddz)
        dx[3] = numpy.sqrt(numpy.mean(numpy.square(dx[:3])))
        mesh_dx[k] = numpy.asarray(dx)
    mesh.close()
    h5py.File(mesh_h5_filename,'a').create_dataset("Mesh/elem_dx", dtype=numpy.float64, data=mesh_dx, compression="gzip")
    return mesh_dx, cells

def compute_local_cfl(ids, h5_files, mesh_dx, cells, dt):
    if len(ids)==1:
        print ('    reading', len(ids), 'file:', ids, h5_files[ids[0]], flush=True)
    else:
        print ('    reading', len(ids), 'files:', ids, h5_files[ids[0]], ' ... ',  h5_files[ids[-1]], flush=True)

    number_of_cells = cells.shape[0]

    # loop over files
    for i in ids:
        hw = h5py.File(h5_files[i], 'r')
        lu = numpy.asarray(hw['/Solution/u'])
		
        cfl = numpy.zeros(number_of_cells, dtype=numpy.float64)
        cfl_nodal = numpy.zeros(len(lu), dtype=numpy.float64)
        for j in range(number_of_cells):
            ptids = cells[j]
            cfl[j] = (numpy.mean(lu[ptids,0]) / mesh_dx[j][0] \
                    + numpy.mean(lu[ptids,1]) / mesh_dx[j][1] \
                    + numpy.mean(lu[ptids,2]) / mesh_dx[j][2])*dt
            #assign a cfl for each point in cell, making sure it is the maximum of all the cells that the point is part of
            for idx in ids:
                if cfl[j]>cfl_nodal[idx]:
                    cfl_nodal[idx]=cfl[j]
        hw.close()
        
        # write to h5
        output_filename = h5_files[i].replace('_up.h5', '_cfl.h5')
        print ('Writing data to %s ...'%(output_filename), end='')
        hf = h5py.File(output_filename, 'w')
        hf.create_dataset("cfl", dtype=numpy.float64, data=cfl_nodal, compression="gzip")
        hf.close()
        print (' done.')

def write_xdmf(folder, mesh_file, period, h5_files):

    file_count = len(h5_files)

    with h5py.File(mesh_file, 'r') as hf:
        points = np.array(hf['Mesh']['coordinates'])
        cells = np.array(hf['Mesh']['topology'])

    xdmffile = folder +'/cfl.xdmf'
    xdmftext =  '''<?xml version="1.0"?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="3.0" xmlns:xi="http://www.w3.org/2001/XInclude">
  <Domain>
    <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">
      <Grid Name="mesh" GridType="Uniform">
        <Topology NumberOfElements="%d" TopologyType="Tetrahedron" NodesPerElement="4">
          <DataItem Dimensions="%d 4" NumberType="UInt" Format="HDF">%s:/Mesh/topology</DataItem>
        </Topology>
        <Geometry GeometryType="XYZ">
          <DataItem Dimensions="%d 3" Format="HDF">%s:/Mesh/coordinates</DataItem>
        </Geometry>
      </Grid>
    </Grid>
'''%(cells.shape[0], cells.shape[0], mesh_file.name, len(points), mesh_file.name)
    f = open(xdmffile, 'w')
    f.write(xdmftext)
    f.close

    xml_node_grid_vector_tmp = '''    <Grid>
      <xi:include xpointer="xpointer(//Grid[@Name=&quot;TimeSeries&quot;]/Grid[1]/*[self::Topology or self::Geometry])" />
      <Time Value="%%f" />
      <Attribute Name="cfl" AttributeType="Vector" Center="Node">
        <DataItem Dimensions="%d 1" Format="HDF">%%s:/cfl</DataItem>
      </Attribute>
    </Grid>
'''%(len(points), len(points))
    f = open(xdmffile, 'a')
    for i in range(file_count):
        cfl_file = h5_files[i].replace('_up.h5', '_cfl.h5')
        time_ = float(cfl_file.stem.split('_t=')[1].split('_ts')[0]) / 1000.0
        text = xml_node_grid_vector_tmp%(time_, cfl_file)
        f.write(text)
    f.close

################################################################
def compute_cfl(input_folder, interval, nproc, period):
    pos = -2 if (input_folder[-1] == '/') else -1
    folder_itself = input_folder.split('/')[pos]
    timesteps = int(folder_itself.split("_ts")[-1].split("_cy")[0])

    mesh_h5_filename, case_folder, case_name = job_utils.get_case_mesh_filename(input_folder)

    if not os.path.exists(mesh_h5_filename):
        print ('No mesh file found: %s \nYou may running this script from an incorrect folder.\n'%mesh_h5_filename)
        exit(1)
    print ('Looking for mesh file:', mesh_h5_filename, ' and loading volume mesh (/Mesh/coordinates, /Mesh/topology).')
    print ('Computing mesh element length scales.')
    mesh_dx, mesh_cells = get_mesh_deltas(mesh_h5_filename)

    print ('Looking inside', input_folder, '...')

    h5_files = glob.glob(input_folder+"/*_up.h5")
    # sort the files according to the simulation time
    h5_files = sorted(h5_files, key=lambda N: int(N.split("ts=")[1].split("_up.h5")[0]))
    h5_files = h5_files[0:len(h5_files):interval]
    file_count = len(h5_files)
    
    write_xdmf(input_folder, mesh_h5_filename, period, h5_files)

    print ('   found', file_count, 'up files for', mesh_cells.shape[0], 'elements.')

    # determine the time increment based on the number of samples and the corresponding FFT frequencies
    dt = period/timesteps
    
    # make group and divide the procedure
    step = max(int(file_count / nproc), 1)
    rng = list(range(0,file_count))
    groups = [rng[i:i+step] for i  in range(rng[0], rng[-1]+1, step)]

    print ('Reading', len(h5_files), 'files in', len(groups), 'of', step, '.')

    p_list=[]
    for i,g in enumerate(groups):
        pr = multiprocessing.Process(target=compute_local_cfl, name='Process'+str(i), args=(g,h5_files,mesh_dx,mesh_cells, dt))
        p_list.append(pr)
    for pr in p_list: pr.start()
    # Wait for all the processes to finish
    for pr in p_list: pr.join()

    print (' done.', flush=True)

if __name__ == '__main__':
	#first argument is the results folder
    ncore = multiprocessing.cpu_count()
    interval = 1
    period = float(sys.argv[2])
    print ( 'Performing CFL computation on %d core%s and interval of %d.'%(ncore,'s' if ncore>1 else '',interval) )
    compute_cfl(sys.argv[1], interval, ncore, period)
