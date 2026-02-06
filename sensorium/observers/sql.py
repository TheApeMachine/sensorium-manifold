import sqlparse


class SQLObserver:
    def __init__(self, sql_query: str):
        self.sql_query = sql_query

    def observe(self, state: dict) -> dict:
        parsed = sqlparse.parse(self.sql_query)
        for statement in parsed:
            for token in statement.tokens:
                if token.ttype == sqlparse.tokens.Name:
                    print(token.value)

            match token.value:
                case "select":
                    print("select")
                case "from":
                    print("from")
                case "where":
                    print("where")
                case _:
                    print(token.value)
        
        return state