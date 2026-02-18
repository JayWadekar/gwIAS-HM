readligo.dq2segs
================

Back to :doc:`Module page <../modules/readligo>`

Summary
-------

This function takes a DQ CHANNEL (as returned by loaddata or getstrain) and the GPS_START time of the channel and returns a segment list. The DQ Channel is assumed to be a 1 Hz channel.

Signature
---------

.. code-block:: python

   def dq2segs(channel, gps_start)

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
   * - ``gps_start``
     - -
     - -
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

   This function takes a DQ CHANNEL (as returned by loaddata or getstrain) and 
   the GPS_START time of the channel and returns a segment
   list.  The DQ Channel is assumed to be a 1 Hz channel.
   
   Returns of a list of segment GPS start and stop times.
