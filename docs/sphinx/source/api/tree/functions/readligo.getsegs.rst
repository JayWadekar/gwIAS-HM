readligo.getsegs
================

Back to :doc:`Module page <../modules/readligo>`

Summary
-------

Method for constructing a segment list from LOSC data files. By default, the method uses files in the current working directory to construct a segment list.

Signature
---------

.. code-block:: python

   def getsegs(start, stop, ifo, flag = 'DATA', filelist = None, strain_chan = None, dq_chan = None, inj_chan = None)

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
   * - ``flag``
     - -
     - 'DATA'
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

   Method for constructing a segment list from 
   LOSC data files.  By default, the method uses
   files in the current working directory to 
   construct a segment list.  
   
   If a FileList is passed in the flag FILELIST,
   then those files will be searched for segments
   passing the DQ flag passed as the FLAG argument.
   
   START is the start time GPS
   STOP is the stop time GPS
   STRAIN_CHAN is the channel name of the strain vector in GWF files.
   DQ_CHAN is the channel name of the data quality vector in GWF files.
   INJ_CHAN is the channel name of the injection vector in GWF files.
