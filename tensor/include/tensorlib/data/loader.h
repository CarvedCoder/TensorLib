#ifndef LOADER_H
#define LOADER_H

#include <csvlib/csv.h>
#include <tensorlib/tensor.h>

namespace Data {
Tensor toTensor(CSVData &csv_data);
} // namespace Data

#endif // !LOADER_H
