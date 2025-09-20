from CQCParse import relay
from CQCParse.utils import PKG_ROOT as CQCPARSE_ROOT
from . import wilson_utils as wu


source_loc = CQCPARSE_ROOT + '/CQCParse/files_examples/calculations.csv'
vault = relay.DataVault(source_loc)
