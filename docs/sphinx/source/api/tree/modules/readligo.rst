readligo
========

Back to :doc:`API tree index <../index>`

Purpose
-------

LIGO frame/HDF5 loading and segment-manipulation helpers.

Module summary
--------------

readligo.py

.. list-table:: Top-level functions
   :header-rows: 1

   * - Function
     - Summary
   * - :doc:`dq2segs <../functions/readligo.dq2segs>`
     - This function takes a DQ CHANNEL (as returned by loaddata or getstrain) and the GPS_START time of the channel and returns a segment list. The DQ Channel is assumed to be a 1 Hz channel.
   * - :doc:`dq_channel_to_seglist <../functions/readligo.dq_channel_to_seglist>`
     - WARNING: This function is designed to work the output of the low level function LOADDATA, not the output from the main data loading function GETSTRAIN.
   * - :doc:`getsegs <../functions/readligo.getsegs>`
     - Method for constructing a segment list from LOSC data files. By default, the method uses files in the current working directory to construct a segment list.
   * - :doc:`getstrain <../functions/readligo.getstrain>`
     - START should be the starting gps time of the data to be loaded. STOP should be the end gps time of the data to be loaded. IFO should be 'H1', 'H2', or 'L1'. FILELIST is an optional argument that is a FileList() instance. STRAIN_CHAN is the channel name of the strain vector in GWF files. DQ_CHAN is the channel name of the data quality vector in GWF files. INJ_CHAN is the channel name of the injection vector in GWF files.
   * - :doc:`loaddata <../functions/readligo.loaddata>`
     - The input filename should be a LOSC .hdf5 file or a LOSC .gwf file. The file type will be determined from the extenstion. The detector should be H1, H2, or L1.
   * - :doc:`read_frame <../functions/readligo.read_frame>`
     - Helper function to read frame files
   * - :doc:`read_hdf5 <../functions/readligo.read_hdf5>`
     - Helper function to read HDF5 files

.. list-table:: Classes
   :header-rows: 1

   * - Class
     - Summary
   * - :doc:`FileList <../classes/readligo.FileList>`
     - Class for lists of LIGO data files.
   * - :doc:`SegmentList <../classes/readligo.SegmentList>`
     - No class docstring summary available.

.. toctree::
   :hidden:
   :maxdepth: 1

   ../classes/readligo.FileList
   ../classes/readligo.SegmentList
   ../functions/readligo.dq2segs
   ../functions/readligo.dq_channel_to_seglist
   ../functions/readligo.getsegs
   ../functions/readligo.getstrain
   ../functions/readligo.loaddata
   ../functions/readligo.read_frame
   ../functions/readligo.read_hdf5
