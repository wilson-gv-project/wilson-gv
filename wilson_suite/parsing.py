from CQCParse import relay
import wilson_utils as wu

source_loc=wu.paths.SUITE_ROOT + '/wilson_intensities/tests/test_database/mini_files_database.csv'
vault = relay.DataVault(source_loc)