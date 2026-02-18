utils.interpolate_wf_fd
=======================

Back to :doc:`Module page <../modules/utils>`

Summary
-------

get log-log interpolant of amp & lin of phase with old_f[0] >= 0,

Signature
---------

.. code-block:: python

   def interpolate_wf_fd(old_f, old_wf, log_in = False)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``old_f``
     - -
     - -
     - (ordered) array of rfft frequencies >= 0 (in Hz) where waveform is saved
   * - ``old_wf``
     - -
     - -
     - len(old_f) array with frequency domain waveform h(f) OR log waveform log(h) = np.log(np.abs(h)) + 1j\\\*np.unwrap(np.angle(h)) => set log_in=True --> BE SURE TO UNWRAP PHASE if log_in !!!
   * - ``log_in``
     - -
     - False
     - bool indicating if old_wf is log(h) => real, imag = amp, unwrapped_phase

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - Function that takes frequencies (Hz) & returns waveform at those frequencies

Docstring
---------

.. code-block:: text

   get log-log interpolant of amp & lin of phase with old_f[0] >= 0,
   :param old_f: (ordered) array of rfft frequencies >= 0 (in Hz) where waveform is saved
   :param old_wf: len(old_f) array with frequency domain waveform h(f)
       OR log waveform log(h) = np.log(np.abs(h)) + 1j*np.unwrap(np.angle(h)) => set log_in=True
       --> BE SURE TO UNWRAP PHASE if log_in !!!
   :param log_in: bool indicating if old_wf is log(h) => real, imag = amp, unwrapped_phase
   :return: Function that takes frequencies (Hz) & returns waveform at those frequencies
