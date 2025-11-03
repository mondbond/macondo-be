# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from guardrails import Guard, Validator
# from guardrails.validator_base import register_validator
# from guardrails.hub import CompetitorCheck
# # from guardrails.hub import RegexMatch
#
#
# from src.llm.llm_provider import get_llm
#
# model = get_llm()
#
# class HateSpeechValidator(Validator):
#     rail_alias = "hate_speech"
#     def validate(self, value, **kwargs):
#         # Simple check for hate speech keywords
#         hate_keywords = ["hate", "kill", "destroy", "violence"]
#         if any(word in value.lower() for word in hate_keywords):
#             return self.fail("Hate speech detected.")
#         return self.pass_()
#
#
#
# competitors_list = ["toyota", "honda", "mazda"]
# guard = Guard().use(
#     RegexMatch(regex="^[A-Z][a-z]*$")
# )
#
# prompt = ChatPromptTemplate.from_template("Answer this question {question}")
# output_parser = StrOutputParser()
#
# chain = prompt | model | guard.to_runnable() | output_parser
#
# result = chain.invoke({"question": "I hate everyone"})
# print(result)
