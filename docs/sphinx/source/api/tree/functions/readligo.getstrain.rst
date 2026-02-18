readligo.getstrain
==================

Back to :doc:`Module page <../modules/readligo>`

Summary
-------

START should be the starting gps time of the data to be loaded. STOP should be the end gps time of the data to be loaded. IFO should be 'H1', 'H2', or 'L1'. FILELIST is an optional argument that is a FileList() instance. STRAIN_CHAN is the channel name of the strain vector in GWF files. DQ_CHAN is the channel name of the data quality vector in GWF files. INJ_CHAN is the channel name of the injection vector in GWF files.

Signature
---------

.. code-block:: python

   def getstrain(start, stop, ifo, filelist = None, strain_chan = None, dq_chan = None, inj_chan = None)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``start``
     - -
     - -
     - -
   * - ``stop``
     - -
     - -
     - -
   * - ``ifo``
     - -
     - -
     - -
   * - ``filelist``
     - -
     - None
     - -
   * - ``strain_chan``
     - -
     - None
     - -
   * - ``dq_chan``
     - -
     - None
     - -
   * - ``inj_chan``
     - -
     - None
     - -

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - -

Docstring
---------

.. code-block:: text

   START should be the starting gps time of the data to be loaded.
   STOP  should be the end gps time of the data to be loaded.
   IFO should be 'H1', 'H2', or 'L1'.
   FILELIST is an optional argument that is a FileList() instance.
   STRAIN_CHAN is the channel name of the strain vector in GWF files.
   DQ_CHAN is the channel name of the data quality vector in GWF files.
   INJ_CHAN is the channel name of the injection vector in GWF files.
   
   The return value is (strain, meta, dq)
   
   STRAIN: The data as a strain time series
   META: A dictionary of meta data, especially the start time, stop time, 
         and sample time
   DQ: A dictionary of the data quality flags
