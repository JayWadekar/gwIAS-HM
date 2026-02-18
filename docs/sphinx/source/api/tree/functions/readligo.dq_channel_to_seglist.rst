readligo.dq_channel_to_seglist
==============================

Back to :doc:`Module page <../modules/readligo>`

Summary
-------

WARNING: This function is designed to work the output of the low level function LOADDATA, not the output from the main data loading function GETSTRAIN.

Signature
---------

.. code-block:: python

   def dq_channel_to_seglist(channel, fs = 4096)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``channel``
     - -
     - -
     - -
   * - ``fs``
     - -
     - 4096
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

   WARNING: 
   This function is designed to work the output of the low level function
   LOADDATA, not the output from the main data loading function GETSTRAIN.
   
   Takes a data quality 1 Hz channel, as returned by
   loaddata, and returns a segment list.  The segment
   list is really a list of slices for the strain 
   associated strain vector.  
   
   If CHANNEL is a dictionary instead of a single channel,
   an attempt is made to return a segment list for the DEFAULT
   channel.  
   
   Returns a list of slices which can be used directly with the 
   strain and time outputs of LOADDATA.
