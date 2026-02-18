python_utils.tukwin_bandpass
============================

Back to :doc:`Module page <../modules/python_utils>`

Summary
-------

taper_width is frequency interval length in Hz to be tapered at each end f_nyq & rfft_len are the nyquist frequency and length of the rfftfreq array

Signature
---------

.. code-block:: python

   def tukwin_bandpass(taper_width, f_nyq = None, rfft_len = None, f_rfft = None)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``taper_width``
     - -
     - -
     - -
   * - ``f_nyq``
     - -
     - None
     - -
   * - ``rfft_len``
     - -
     - None
     - -
   * - ``f_rfft``
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

   taper_width is frequency interval length in Hz to be tapered at each end
   f_nyq & rfft_len are the nyquist frequency and length of the rfftfreq array
