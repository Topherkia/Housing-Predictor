class HousingAssistant:

    def __init__(
        self,
        xgboost_model,
        qwen,
        rag
    ):

        self.xgboost = xgboost_model
        self.qwen = qwen
        self.rag = rag

