triggers_single_detector_HM.TriggerList.process_clist
=====================================================

Back to :doc:`Class page <../classes/triggers_single_detector_HM.TriggerList>`

Summary
-------

Applies correction factor to times and SNRs and makes real array clist is assumed to have the following columns: if self.save_hole_correction is True: time, (overlap.real, overlap.imag) x 3, hole_correction x 3, psd_drift_correction, basis_coeff1, basis_coeff2, ... else: time, (overlap.real, overlap.imag) x 3, psd_drift_correction, basis_coeff1, basis_coeff2, ...

Signature
---------

.. code-block:: python

   def process_clist(self, clist, adjust_times = True)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``clist``
     - -
     - -
     - n_trigger x (4/5 + n_basis coeff) array with non-shift-corrected time, non-renormalized cos + i sin overlaps, psd drift correction, basis coefficients (can be vector for n_trigger = 1)
   * - ``adjust_times``
     - -
     - True
     - If True, we adjust times to assign triggers to the linear-free time of the calpha waveforms

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - n_triggers x (6/7 + basis coeffs) array with time, total SNR^2, (normfac/constant bringing score to amplitude units), (optional hole correction), psd_drift_correction, cosine SNR, sine SNR, and basis coefficients (can be vector for n_trigger = 1)

Docstring
---------

.. code-block:: text

   Applies correction factor to times and SNRs and makes real array
   clist is assumed to have the following columns:
   if self.save_hole_correction is True:
   time, (overlap.real, overlap.imag) x 3, hole_correction x 3,
   psd_drift_correction, basis_coeff1, basis_coeff2, ...
   else:
   time, (overlap.real, overlap.imag) x 3, psd_drift_correction,
   basis_coeff1, basis_coeff2, ...
   :param clist: n_trigger x (4/5 + n_basis coeff) array with
                 non-shift-corrected time, non-renormalized cos + i sin
                 overlaps, psd drift correction, basis coefficients
                 (can be vector for n_trigger = 1)
   :param adjust_times:
       If True, we adjust times to assign triggers to the linear-free time
       of the calpha waveforms
   :return: n_triggers x (6/7 + basis coeffs) array with time, total SNR^2,
            (normfac/constant bringing score to amplitude units),
            (optional hole correction), psd_drift_correction, cosine SNR,
            sine SNR, and basis coefficients
            (can be vector for n_trigger = 1)
