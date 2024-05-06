#pragma once

#include <string>

#include "models.grpc.pb.h"

namespace llm {

class ModelsHandler : public proto::Models::Service {
 public:
  ModelsHandler(const std::string& model_id);

  grpc::Status List(grpc::ServerContext* context,
                    const proto::ListRequest* request,
                    proto::ListResponse* response) override;

 private:
  std::string model_id_;
  // model created time
  // TODO: read from model config
  uint32_t created_;
};

}  // namespace llm