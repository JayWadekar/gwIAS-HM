utils.create_chirp_mass_directory_dict
======================================

Back to :doc:`Module page <../modules/utils>`

Summary
-------

Reads the names of all subdirectories, and creates a dictionary of dictionaries, with the outer dictionary having chirp mass ids as keys and the inner dictionary having subbank ids as keys and subdirectory names as values

Signature
---------

.. code-block:: python

   def create_chirp_mass_directory_dict(base_path, prefix, suffix)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``base_path``
     - -
     - -
     - Base path to look for subdirectories
   * - ``prefix``
     - -
     - -
     - Prefix to match the subdirectories for the given run and source type, can have wildcards (see BBH_PREFIXES etc for examples). The prefix should match the part of the subdirectory name before the chirp mass id if the chirp mass id is absent, it makes it zero (like O1 BNS) The part after the chirp mass id is supposed to end with the subbank id
   * - ``suffix``
     - -
     - -
     - Any desired suffix to demand at the end of the subdirectories' names (cand type is the use case). Can be the empty string

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

   Reads the names of all subdirectories, and creates a dictionary of
   dictionaries, with the outer dictionary having chirp mass ids as keys and
   the inner dictionary having subbank ids as keys and subdirectory names as
   values
   :param base_path: Base path to look for subdirectories
   :param prefix:
       Prefix to match the subdirectories for the given run and source type,
       can have wildcards (see BBH_PREFIXES etc for examples). The prefix
       should match the part of the subdirectory name before the chirp mass id
       if the chirp mass id is absent, it makes it zero (like O1 BNS)
       The part after the chirp mass id is supposed to end with the subbank id
   :param suffix:
       Any desired suffix to demand at the end of the subdirectories' names
       (cand type is the use case). Can be the empty string
