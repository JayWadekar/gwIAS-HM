ranking_HM.Rank.from_hdf5_static
================================

Back to :doc:`Class page <../classes/ranking_HM.Rank>`

Summary
-------

Load attributes from a hdf5 file into an instance of the class

Signature
---------

.. code-block:: python

   def from_hdf5_static(instance, path, outformat = h5py.Dataset, mode = 'r', load_example_trigs = True, outputdir_to_use = None)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``instance``
     - -
     - -
     - Instance of the class to load the attributes into
   * - ``path``
     - -
     - -
     - Path to the hdf5 file, can also be a File object
   * - ``outformat``
     - -
     - h5py.Dataset
     - Format of the big data in the output (default is dataset, can also be mmap). It can be a string or a type
   * - ``mode``
     - -
     - 'r'
     - Mode to open the hdf5 file or mmap in
   * - ``load_example_trigs``
     - -
     - True
     - Boolean flag to load the example trigs
   * - ``outputdir_to_use``
     - -
     - None
     - If given, replace the outputdir with this path for the example trigs

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - BankBG instance with self.fobj set to the hdf5 file

Docstring
---------

.. code-block:: text

   Load attributes from a hdf5 file into an instance of the class
   :param instance: Instance of the class to load the attributes into
   :param path: Path to the hdf5 file, can also be a File object
   :param outformat:
       Format of the big data in the output (default is dataset, can also
       be mmap). It can be a string or a type
   :param mode: Mode to open the hdf5 file or mmap in
   :param load_example_trigs: Boolean flag to load the example trigs
   :param outputdir_to_use:
       If given, replace the outputdir with this path for the example trigs
   :return: BankBG instance with self.fobj set to the hdf5 file
