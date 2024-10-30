'''
This script is used to run the model to infer proofs for problems, theorem statements or lemmas written in Isabelle/HOL.
Usage:
    python isabelle-proof-generator/stages/3_run_model.py <model_name> <device mode>
    - model_name: The name of the model to use for inference.

    - device mode: The device to use for inference. It can be cpu, cuda, half or low.
        - cpu: infer by cpu
        - cuda: infer by gpu
        - half: infer by gpu using half precision
        - low: infer by gpu using low cpu memory usage
'''

from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch
import sys

PROMPT_TEMPLATE_QUESTION_ANSWER = 'You are now an specialized agent to infer proofs for problems, theorem statements or lemmas written in Isabelle/HOL. You are going to receive instructions of what you need to infer, and you will also receive some context and the corresponding problem, theorem statement or lemma. When you answer, please do it reasoning step by step.'
PROMPT_TEMPLATE_QUESTION_ANSWER_WITH_CONTEXT = 'You are now an specialized agent to infer proofs for problems, theorem statements or lemmas written in Isabelle/HOL. You are going to receive instructions of what you need to infer, and you will also receive some context and the corresponding problem, theorem statement or lemma. When you answer, please do it reasoning step by step.'

'''
This function is used to stream the generated text from the model.
'''


def stream(fullprompt, device, initial_max_tokens=200, continuation_tokens=100):
    inputs = tokenizer([fullprompt], return_tensors="pt").to(device)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generated_text = ""
    while True:
        # outputs = model.generate(**inputs, streamer=streamer, max_new_tokens=initial_max_tokens)
        outputs = model.generate(
            **inputs, 
            streamer=streamer, 
            max_new_tokens=initial_max_tokens,
            pad_token_id=tokenizer.eos_token_id
        )

        generated_text += tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if outputs[0][-1] == tokenizer.eos_token_id:
            break
        
        # If we did not reach the EOS token, continue generating
        inputs = tokenizer([generated_text], return_tensors="pt").to(device)
        initial_max_tokens = continuation_tokens

    return generated_text


def infer_proof(context, theorem_statement, device):
    system_prompt = PROMPT_TEMPLATE_QUESTION_ANSWER_WITH_CONTEXT if context else PROMPT_TEMPLATE_QUESTION_ANSWER
    B_INST, E_INST = f"[INST]Given the problem context {context}, " if context else "[INST]", "[/INST]"

    fullprompt = f"{system_prompt}{B_INST}Infer a proof for the following Isabelle/HOL theorem statement/s: {theorem_statement.strip()}\n{E_INST}"
    print("Full prompt:\n")
    print(fullprompt)
    print('Infering proof...\n')
    stream(fullprompt, device)

'''
This function is used to infer a proof for a given theorem statement.
It works as follows:
- It composes the full prompt with the theorem statement and the context.
'''
def infer_proof(context, theorem_statement, device):
    system_prompt = PROMPT_TEMPLATE_QUESTION_ANSWER_WITH_CONTEXT if context else PROMPT_TEMPLATE_QUESTION_ANSWER
    B_INST, E_INST = f"[INST]Given the problem context {context}, " if context else "[INST]", "[/INST]"

    fullprompt = f"{system_prompt}{B_INST}Infer a proof for the following Isabelle/HOL theorem statement/s: {theorem_statement.strip()}\n{E_INST}"
    print("Full prompt:\n")
    print(fullprompt)
    print('Infering proof...\n')
    stream(fullprompt, device)

model_name = sys.argv[1]
requested_device = sys.argv[2]

if requested_device == "cpu": device = "cpu"
elif requested_device == "cuda": device = "cuda" if torch.cuda.is_available() else "cpu"
elif requested_device == "half": device = "cuda" if torch.cuda.is_available() else "cpu"

torch.cuda.empty_cache()
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_name)

if requested_device == "low":
    model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True)

else: model = AutoModelForCausalLM.from_pretrained(model_name)

if requested_device == "half": model = model.half()
elif requested_device == "cuda": model = model.to(device)

problems = [
    {context: '''We define the SLin I/O-automaton and prove its composition property. The SLin I/O-automaton is at the heart of the Speculative Linearizability framework. This framework simplifies devising robust distributed algorithms by enabling their decomposition into independent modules that can be analyzed and proved correct in isolation. It is particularly useful when working in a distributed environment, where the need to tolerate faults and asynchrony has made current monolithic protocols so intricate that it is no longer tractable to check their correctness. Our theory contains a formalization of simulation proof techniques in the I/O-automata of Lynch and Tuttle and a typical example of a refinement proof.''', theorem_statement: '''lemma internal_inter_external: assumes "is_asig sig" shows "internals sig \<inter> externals sig = {}" using assms by (auto simp add:internals_def externals_def is_asig_def) definition hide_asig where "hide_asig asig actns \<equiv> \<lparr>inputs = inputs asig - actns, outputs = outputs asig - actns, internals = internals asig \<union>actns\<rparr>" end subsection \<open>I/O Automata\<close> type_synonym ('s,'a) transition = "'s \<times> 'a \<times> 's" record ('s,'a) ioa = asig::"'a signature" start::"'s set" trans::"('s,'a)transition set" context IOA begin abbreviation "act A \<equiv> actions (asig A)" abbreviation "ext A \<equiv> externals (asig A)" abbreviation int where "int A \<equiv> internals (asig A)" abbreviation "inp A \<equiv> inputs (asig A)" abbreviation "out A \<equiv> outputs (asig A)" abbreviation "local A \<equiv> locals (asig A)" definition is_ioa::"('s,'a) ioa \<Rightarrow> bool" where "is_ioa A \<equiv> is_asig (asig A) \<and> (\<forall> triple \<in> trans A . (fst o snd) triple \<in> act A)" definition hide where "hide A actns \<equiv> A\<lparr>asig := hide_asig (asig A) actns\<rparr>" definition is_trans::"'s \<Rightarrow> 'a \<Rightarrow> ('s,'a)ioa \<Rightarrow> 's \<Rightarrow> bool" where "is_trans s1 a A s2 \<equiv> (s1,a,s2) \<in> trans A" notation is_trans ("_ \<midarrow>_\<midarrow>_\<longrightarrow> _" [81,81,81,81] 100) definition rename_set where "rename_set A ren \<equiv> {b. \<exists> x \<in> A . ren b = Some x}" definition rename where "rename A ren \<equiv> \<lparr>asig = \<lparr>inputs = rename_set (inp A) ren, outputs = rename_set (out A) ren, internals = rename_set (int A) ren\<rparr>, start = start A, trans = {tr. \<exists> x . ren (fst (snd tr)) = Some x \<and> (fst tr) \<midarrow>x\<midarrow>A\<longrightarrow> (snd (snd tr))}\<rparr>" text \<open>Reachable states and invariants\<close> inductive reachable :: "('s,'a) ioa \<Rightarrow> 's \<Rightarrow> bool" for A :: "('s,'a) ioa" where reachable_0: "s \<in> start A \<Longrightarrow> reachable A s" | reachable_n: "\<lbrakk> reachable A s; s \<midarrow>a\<midarrow>A\<longrightarrow> t \<rbrakk> \<Longrightarrow> reachable A t" definition invariant where "invariant A P \<equiv> (\<forall> s . reachable A s \<longrightarrow> P(s))" theorem invariantI: fixes A P assumes "\<And> s . s \<in> start A \<Longrightarrow> P s" and "\<And> s t a . \<lbrakk>reachable A s; P s; s \<midarrow>a\<midarrow>A\<longrightarrow> t\<rbrakk> \<Longrightarrow> P t" shows "invariant A P" proof - { fix s assume "reachable A s" hence "P s" proof (induct rule:reachable.induct) fix s assume "s \<in> start A" thus "P s" using assms(1) by simp next fix a s t assume "reachable A s" and "P s" and " s \<midarrow>a\<midarrow>A\<longrightarrow> t" thus "P t" using assms(2) by simp qed } thus ?thesis by (simp add:invariant_def) qed end subsection \<open>Composition of Families of I/O Automata\<close> record ('id, 'a) family = ids :: "'id set" memb :: "'id \<Rightarrow> 'a" context IOA begin definition is_ioa_fam where "is_ioa_fam fam \<equiv> \<forall> i \<in> ids fam . is_ioa (memb fam i)" definition compatible2 where "compatible2 A B \<equiv> out A \<inter> out B = {} \<and> int A \<inter> act B = {} \<and> int B \<inter> act A = {}" definition compatible::"('id, ('s,'a)ioa) family \<Rightarrow> bool" where "compatible fam \<equiv> finite (ids fam) \<and> (\<forall> i \<in> ids fam . \<forall> j \<in> ids fam . i \<noteq> j \<longrightarrow> compatible2 (memb fam i) (memb fam j))" definition asig_comp2 where "asig_comp2 A B \<equiv> \<lparr>inputs = (inputs A \<union> inputs B) - (outputs A \<union> outputs B), outputs = outputs A \<union> outputs B, internals = internals A \<union> internals B\<rparr>" definition asig_comp::"('id, ('s,'a)ioa) family \<Rightarrow> 'a signature" where "asig_comp fam \<equiv> \<lparr> inputs = \<Union>i\<in>(ids fam). inp (memb fam i) - (\<Union>i\<in>(ids fam). out (memb fam i)), outputs = \<Union>i\<in>(ids fam). out (memb fam i), internals = \<Union>i\<in>(ids fam). int (memb fam i) \<rparr>" definition par2 (infixr "\<parallel>" 10) where "A \<parallel> B \<equiv> \<lparr>asig = asig_comp2 (asig A) (asig B), start = {pr. fst pr \<in> start A \<and> snd pr \<in> start B}, trans = {tr. let s = fst tr; a = fst (snd tr); t = snd (snd tr) in (a \<in> act A \<or> a \<in> act B) \<and> (if a \<in> act A then fst s \<midarrow>a\<midarrow>A\<longrightarrow> fst t else fst s = fst t) \<and> (if a \<in> act B then snd s \<midarrow>a\<midarrow>B\<longrightarrow> snd t else snd s = snd t) }\<rparr>" definition par::"('id, ('s,'a)ioa) family \<Rightarrow> ('id \<Rightarrow> 's,'a)ioa" where "par fam \<equiv> let ids = ids fam; memb = memb fam in \<lparr> asig = asig_comp fam, start = {s . \<forall> i\<in>ids . s i \<in> start (memb i)}, trans = { (s, a, s') . (\<exists> i\<in>ids . a \<in> act (memb i)) \<and> (\<forall> i\<in>ids . if a \<in> act (memb i) then s i \<midarrow>a\<midarrow>(memb i)\<longrightarrow> s' i else s i = (s' i)) } \<rparr>" lemmas asig_simps = hide_asig_def is_asig_def locals_def externals_def actions_def hide_def compatible_def asig_comp_def lemmas ioa_simps = rename_def rename_set_def is_trans_def is_ioa_def par_def end subsection \<open>Executions and Traces\<close> type_synonym ('s,'a)pairs = "('a\<times>'s) list" type_synonym ('s,'a)execution = "'s \<times> ('s,'a)pairs" type_synonym 'a trace = "'a list" record ('s,'a)execution_module = execs::"('s,'a)execution set" asig::"'a signature" record 'a trace_module = traces::"'a trace set" asig::"'a signature" context IOA begin fun is_exec_frag_of::"('s,'a)ioa \<Rightarrow> ('s,'a)execution \<Rightarrow> bool" where "is_exec_frag_of A (s,(ps#p')#p) = (snd p' \<midarrow>fst p\<midarrow>A\<longrightarrow> snd p \<and> is_exec_frag_of A (s, (ps#p')))" | "is_exec_frag_of A (s, [p]) = s \<midarrow>fst p\<midarrow>A\<longrightarrow> snd p" | "is_exec_frag_of A (s, []) = True" definition is_exec_of::"('s,'a)ioa \<Rightarrow> ('s,'a)execution \<Rightarrow> bool" where "is_exec_of A e \<equiv> fst e \<in> start A \<and> is_exec_frag_of A e" definition filter_act where "filter_act \<equiv> map fst" definition schedule where "schedule \<equiv> filter_act o snd" definition trace where "trace sig \<equiv> filter (\<lambda> a . a \<in> externals sig) o schedule" definition is_schedule_of where "is_schedule_of A sch \<equiv> (\<exists> e . is_exec_of A e \<and> sch = filter_act (snd e))" definition is_trace_of where "is_trace_of A tr \<equiv> (\<exists> sch . is_schedule_of A sch \<and> tr = filter (\<lambda> a. a \<in> ext A) sch)" definition traces where "traces A \<equiv> {tr. is_trace_of A tr}" lemma traces_alt: shows "traces A = {tr . \<exists> e . is_exec_of A e \<and> tr = trace (ioa.asig A) e}" proof - { fix t assume a:"t \<in> traces A" have "\<exists> e . is_exec_of A e \<and> trace (ioa.asig A) e = t" proof - from a obtain sch where 1:"is_schedule_of A sch" and 2:"t = filter (\<lambda> a. a \<in> ext A) sch" by (auto simp add:traces_def is_trace_of_def) from 1 obtain e where 3:"is_exec_of A e" and 4:"sch = filter_act (snd e)" by (auto simp add:is_schedule_of_def) from 4 and 2 have "trace (ioa.asig A) e = t" by (simp add:trace_def schedule_def) with 3 show ?thesis by fast qed } moreover { fix e assume "is_exec_of A e" hence "trace (ioa.asig A) e \<in> traces A" by (force simp add:trace_def schedule_def traces_def is_trace_of_def is_schedule_of_def is_exec_of_def) } ultimately show ?thesis by blast qed lemmas trace_simps = traces_def is_trace_of_def is_schedule_of_def filter_act_def is_exec_of_def trace_def schedule_def definition proj_trace::"'a trace \<Rightarrow> ('a signature) \<Rightarrow> 'a trace" (infixr "\<bar>" 12) where "proj_trace t sig \<equiv> filter (\<lambda> a . a \<in> actions sig) t" definition ioa_implements :: "('s1,'a)ioa \<Rightarrow> ('s2,'a)ioa \<Rightarrow> bool" (infixr "=<|" 12) where "A =<| B \<equiv> inp A = inp B \<and> out A = out B \<and> traces A \<subseteq> traces B" subsection \<open>Operations on Executions\<close> definition cons_exec where "cons_exec e p \<equiv> (fst e, (snd e)#p)" definition append_exec where "append_exec e e' \<equiv> (fst e, (snd e)@(snd e'))" fun last_state where "last_state (s,[]) = s" | "last_state (s,ps#p) = snd p" lemma last_state_reachable: fixes A e assumes "is_exec_of A e" shows "reachable A (last_state e)" using assms proof - have "is_exec_of A e \<Longrightarrow> reachable A (last_state e)" proof (induction "snd e" arbitrary: e) case Nil from Nil.prems have 1:"fst e \<in> start A" by (simp add:is_exec_of_def) from Nil.hyps have 2:"last_state e = fst e" by (metis last_state.simps(1) surjective_pairing) from 1 and 2 and Nil.hyps show ?case by (metis reachable_0) next case (Cons p ps e) let ?e' = "(fst e, ps)" have ih:"reachable A (last_state ?e')" proof - from Cons.prems and Cons.hyps(2) have "is_exec_of A ?e'" by (simp add:is_exec_of_def) (metis (full_types) IOA.is_exec_frag_of.simps(1) IOA.is_exec_frag_of.simps(3) neq_Nil_conv prod.collapse) with Cons.hyps(1) show ?thesis by auto qed from Cons.prems and Cons.hyps(2) have "(last_state ?e')\<midarrow>(fst p)\<midarrow>A\<longrightarrow>(snd p)" by (simp add:is_exec_of_def) (cases "(A,fst e,ps#p)" rule:is_exec_frag_of.cases, auto) with ih and Cons.hyps(2) show ?case by (metis last_state.simps(2) reachable.simps surjective_pairing) qed thus ?thesis using assms by fastforce qed lemma trans_from_last_state: assumes "is_exec_frag_of A e" and "(last_state e)\<midarrow>a\<midarrow>A\<longrightarrow>s'" shows "is_exec_frag_of A (cons_exec e (a,s'))" using assms by (cases "(A, fst e, snd e)" rule:is_exec_frag_of.cases, auto simp add:cons_exec_def) lemma exec_frag_prefix: fixes A p ps assumes "is_exec_frag_of A (cons_exec e p)" shows "is_exec_frag_of A e" using assms by (cases "(A, fst e, snd e)" rule:is_exec_frag_of.cases, auto simp add:cons_exec_def) lemma trace_same_ext: fixes A B e assumes "ext A = ext B" shows "trace (ioa.asig A) e = trace (ioa.asig B) e" using assms by (auto simp add:trace_def) lemma trace_append_is_append_trace: fixes e e' sig shows "trace sig (append_exec e' e) = trace sig e' @ trace sig e" by (simp add:append_exec_def trace_def schedule_def filter_act_def) lemma append_exec_frags_is_exec_frag: fixes e e' A as assumes "is_exec_frag_of A e" and "last_state e = fst e'" and "is_exec_frag_of A e'" shows "is_exec_frag_of A (append_exec e e')" proof - from assms show ?thesis proof (induct "(fst e',snd e')" arbitrary:e' rule:is_exec_frag_of.induct) case (3 A) from "3.hyps" and "3.prems"(1) show ?case by (simp add:append_exec_def) next case (2 A p) have "last_state e \<midarrow>(fst p)\<midarrow>A\<longrightarrow> snd p" using "2.prems"(2,3) and "2.hyps" by (metis is_exec_frag_of.simps(2) prod.collapse) hence "is_exec_frag_of A (fst e, (snd e)#p)" using "2.prems"(1) by (metis cons_exec_def prod.collapse trans_from_last_state) moreover have "append_exec e e' = (fst e, (snd e)#p)" using "2.hyps" by (metis append_Cons append_Nil append_exec_def) ultimately show ?case by auto next case (1 A ps p' p e') have "is_exec_frag_of A (fst e, (snd e)@((ps#p')#p))" proof - have "is_exec_frag_of A (fst e, (snd e)@(ps#p'))" by (metis "1.hyps" "1.prems" append_exec_def cons_exec_def exec_frag_prefix fst_conv prod_eqI snd_conv) moreover have "snd p' \<midarrow>(fst p)\<midarrow>A\<longrightarrow> snd p" using "1.prems"(3) "1.hyps"(2) by (metis is_exec_frag_of.simps(1) prod.collapse) ultimately show ?thesis by simp qed moreover have "append_exec e e' = (fst e, (snd e)@((ps#p')#p))" by (metis "1.hyps"(2) append_exec_def) ultimately show ?case by simp qed qed'''}
]

for problem in problems:
    context = problem['context']
    theorem_statement = problem['theorem_statement']
    proof = infer_proof(context, theorem_statement, device)
    print('Inferred proof:\n')
    print(proof)