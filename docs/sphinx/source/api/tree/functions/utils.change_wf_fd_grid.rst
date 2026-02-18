utils.change_wf_fd_grid
=======================

Back to :doc:`Module page <../modules/utils>`

Summary
-------

-

Signature
---------

.. code-block:: python

   def change_wf_fd_grid(wf_fd, fs_in, fs_out, pad_mode = 'center')

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``wf_fd``
     - -
     - -
     - FD waveform at frequencies in fs_in
   * - ``fs_in``
     - -
     - -
     - np.fft.rfftfreq(n=nfft_in, d=dt_in)[-keeplen:] where keeplen represents a possible low-frequency cutoff, and dt_in = 1 / (2\\\*fs_in[-1]), nfft_in = T_in / dt_in, with T_in = 1 / df_in = 1 / (fs_in[1] - fs_in[0])
   * - ``fs_out``
     - -
     - -
     - np.fft.rfftfreq(n=nfft_out, d=dt_out)[-getlen:] where getlen represents a possible low-frequency cutoff and nfft_out, dt_out are defined from fs_out the same way as nfft_in, dt_in
   * - ``pad_mode``
     - -
     - 'center'
     - str (\\\`center\\\`, \\\`left\\\`, \\\`right\\\`) specifying where in the TD grid is safe to pad, i.e., location of the interval of zeros connecting the end of the merger signal back to the start of the inspiral

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

   :param wf_fd: FD waveform at frequencies in fs_in
   :param fs_in: np.fft.rfftfreq(n=nfft_in, d=dt_in)[-keeplen:]
       where keeplen represents a possible low-frequency cutoff,
       and dt_in = 1 / (2*fs_in[-1]), nfft_in = T_in / dt_in,
       with T_in = 1 / df_in = 1 / (fs_in[1] - fs_in[0])
   :param fs_out: np.fft.rfftfreq(n=nfft_out, d=dt_out)[-getlen:]
       where getlen represents a possible low-frequency cutoff
       and nfft_out, dt_out are defined from fs_out the same
       way as nfft_in, dt_in
   :param pad_mode: str (`center`, `left`, `right`) specifying
       where in the TD grid is safe to pad, i.e., location of
       the interval of zeros connecting the end of the merger
       signal back to the start of the inspiral
