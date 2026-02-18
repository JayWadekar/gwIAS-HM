readligo.loaddata
=================

Back to :doc:`Module page <../modules/readligo>`

Summary
-------

The input filename should be a LOSC .hdf5 file or a LOSC .gwf file. The file type will be determined from the extenstion. The detector should be H1, H2, or L1.

Signature
---------

.. code-block:: python

   def loaddata(filename, ifo = None, tvec = True, readstrain = True, strain_chan = None, dq_chan = None, inj_chan = None)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``filename``
     - -
     - -
     - -
   * - ``ifo``
     - -
     - None
     - -
   * - ``tvec``
     - -
     - True
     - -
   * - ``readstrain``
     - -
     - True
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

   The input filename should be a LOSC .hdf5 file or a LOSC .gwf
   file.  The file type will be determined from the extenstion.  
   The detector should be H1, H2, or L1.
   
   The return value is: 
   STRAIN, TIME, CHANNEL_DICT
   
   STRAIN is a vector of strain values
   TIME is a vector of time values to match the STRAIN vector
        unless the flag tvec=False.  In that case, TIME is a
        dictionary of meta values.
   CHANNEL_DICT is a dictionary of data quality channels    
   STRAIN_CHAN is the channel name of the strain vector in GWF files.
   DQ_CHAN is the channel name of the data quality vector in GWF files.
   INJ_CHAN is the channel name of the injection vector in GWF files.
