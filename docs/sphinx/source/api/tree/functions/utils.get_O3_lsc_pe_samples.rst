utils.get_O3_lsc_pe_samples
===========================

Back to :doc:`Module page <../modules/utils>`

Summary
-------

Returns a PESummary object to interact with the LSC PE samples

Signature
---------

.. code-block:: python

   def get_O3_lsc_pe_samples(evname, root = LSC_PE_DIR, comoving = True)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``evname``
     - -
     - -
     - Name of the event
   * - ``root``
     - -
     - LSC_PE_DIR
     - Path to directory with the samples of all O3 events
   * - ``comoving``
     - -
     - True
     - Flag whether to load samples labeled by 'comoving'

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - 1. PEsummary object with posterior samples 2. Numpy recarray with prior samples The list of approximants is in object.samples_dict.keys(), and for each approximant, the samples are in object.samples_dict['approximant'] The PSD is in object.psd['approximant'] See https://dcc.ligo.org/public/0169/P2000223/005/PEDataReleaseExample.html

Docstring
---------

.. code-block:: text

   Returns a PESummary object to interact with the LSC PE samples
   :param evname: Name of the event
   :param root: Path to directory with the samples of all O3 events
   :param comoving: Flag whether to load samples labeled by 'comoving'
   :return:
       1. PEsummary object with posterior samples
       2. Numpy recarray with prior samples
       The list of approximants is in object.samples_dict.keys(), and for each
       approximant, the samples are in object.samples_dict['approximant']
       The PSD is in object.psd['approximant']
       See https://dcc.ligo.org/public/0169/P2000223/005/PEDataReleaseExample.html
