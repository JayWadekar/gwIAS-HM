data_operations.calculate_excess_power
======================================

Back to :doc:`Module page <../modules/data_operations>`

Summary
-------

Returns samples of excess power with and without a rolling average

Signature
---------

.. code-block:: python

   def calculate_excess_power(strain_wt, qmask, valid_mask, dt, interval, frng, edgesafety = 1, freqs_lines = None, mask_freqs_in = None, fmax = params.FMAX_OVERLAP, verbose = True, **spec_kwargs)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``strain_wt``
     - -
     - -
     - Whitened strain data
   * - ``qmask``
     - -
     - -
     - Boolean mask with zeros at holes in unwhitened data
   * - ``valid_mask``
     - -
     - -
     - Boolean mask with zeros where we cannot trust whitened data
   * - ``dt``
     - -
     - -
     - Time interval between successive elements of data (s)
   * - ``interval``
     - -
     - -
     - Time interval to look for excess power on (s)
   * - ``frng``
     - -
     - -
     - Frequency interval to look for excess power in (Hz). Pass [0, np.inf] for total power
   * - ``edgesafety``
     - -
     - 1
     - Safety margin at the edge where we cannot trust the nature of the whitened data (we haven't applied this to the mask when we get here)
   * - ``freqs_lines``
     - -
     - None
     - Array with frequencies on which we previously detected lines (optional)
   * - ``mask_freqs_in``
     - -
     - None
     - Boolean mask on freqs_lines with zeros at previously detected varying lines (optional)
   * - ``fmax``
     - -
     - params.FMAX_OVERLAP
     - Maximum frequency involved in the analysis
   * - ``verbose``
     - -
     - True
     - If true, would print details on power detector
   * - ``\*\*spec_kwargs``
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
     - Zeros qmask in place, and returns list of indices that jumped

Docstring
---------

.. code-block:: text

   Returns samples of excess power with and without a rolling average
   :param strain_wt: Whitened strain data
   :param qmask: Boolean mask with zeros at holes in unwhitened data
   :param valid_mask:
       Boolean mask with zeros where we cannot trust whitened data
   :param dt: Time interval between successive elements of data (s)
   :param interval: Time interval to look for excess power on (s)
   :param frng: Frequency interval to look for excess power in (Hz).
                Pass [0, np.inf] for total power
   :param edgesafety:
       Safety margin at the edge where we cannot trust the nature of the
       whitened data (we haven't applied this to the mask when we get here)
   :param freqs_lines:
       Array with frequencies on which we previously detected lines (optional)
   :param mask_freqs_in:
       Boolean mask on freqs_lines with zeros at previously detected varying
       lines (optional)
   :param fmax: Maximum frequency involved in the analysis
   :param verbose: If true, would print details on power detector
   :return: Zeros qmask in place, and returns list of indices that jumped
