import idaapi
import ida_hexrays
import ida_funcs
import idautils
import idc
import ida_bytes
import ida_xref
import ida_ua
import ida_lines
import ida_segment
import ida_name
import json
import jsonlines
import sys
import os
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SuperEnhancedFunctionAnalyzer:
    def __init__(self):
        self.min_func_size = 10
        self.context_lines = 8  # 增加上下文行数
        self.max_call_depth = 5  # 增加调用链深度
        self.min_string_length = 3
        self.max_callers = 50  # 最大调用者数量
        self.arg_search_range = 30  # 参数搜索范围
        # Global Symbol Table: Address (hex string) -> Resolved Name
        # In a real propagation loop, this would be updated iteratively.
        # Here we initialize it with known symbols if available, or keep it ready for updates.
        self.global_symbol_table = {} 

    def update_symbol_table(self, func_ea, name):
        """Update the global symbol table with a resolved name"""
        ea_hex = hex(func_ea)
        self.global_symbol_table[ea_hex] = name
        logger.info(f"Updated Symbol Table: {ea_hex} -> {name}")

    def get_resolved_name(self, func_ea):
        """Retrieve a resolved name from the symbol table, or return None"""
        return self.global_symbol_table.get(hex(func_ea))

        
    def wait_for_analysis(self):
        """等待IDA分析完成"""
        logger.info("Waiting for IDA analysis to complete...")
        idaapi.auto_wait()
        logger.info("Analysis complete")
    
    def get_function_name(self, ea):
        """获取函数名称"""
        name = idc.get_func_name(ea)
        if not name or name.startswith("sub_"):
            name = idc.get_name(ea)
        if not name:
            name = f"sub_{ea:X}"
        return name
    
    def decompile_function(self, ea):
        """反编译函数并优化输出，去除局部变量声明"""
        try:
            if ida_hexrays.init_hexrays_plugin():
                cfunc = ida_hexrays.decompile(ea)
                if cfunc:
                    decompiled = str(cfunc)
                    # 优化反编译代码：去除局部变量声明，保留核心逻辑
                    cleaned_code = self.clean_decompiled_code(decompiled)
                    return cleaned_code
        except Exception as e:
            logger.debug(f"Decompilation failed for {ea:X}: {e}")
        
        # 如果反编译失败，返回汇编代码
        func = idaapi.get_func(ea)
        if not func:
            return ""
            
        asm_lines = []
        current_ea = func.start_ea
        while current_ea < func.end_ea:
            disasm = idc.GetDisasm(current_ea)
            if disasm:
                asm_lines.append(f"{current_ea:08X}: {disasm}")
            current_ea = idc.next_head(current_ea, func.end_ea)
        
        return "\n".join(asm_lines)
    
    def clean_decompiled_code(self, code):
        """清理反编译代码，去除局部变量声明，保留核心逻辑"""
        if not code:
            return code
        
        lines = code.split('\n')
        cleaned_lines = []
        in_var_declaration = False
        brace_count = 0
        
        for line in lines:
            stripped = line.strip()
            
            # 跳过空行
            if not stripped:
                continue
            
            # 检测函数开始的大括号
            if '{' in line:
                brace_count += line.count('{')
                if brace_count == 1 and not in_var_declaration:
                    in_var_declaration = True
                    cleaned_lines.append(line)
                    continue
            
            if '}' in line:
                brace_count -= line.count('}')
                in_var_declaration = False
                cleaned_lines.append(line)
                continue
            
            # 在函数体内，跳过局部变量声明
            if in_var_declaration and brace_count == 1:
                # 识别局部变量声明模式
                var_patterns = [
                    r'^\s*(int|long|short|char|unsigned|signed|void|bool|float|double|size_t|__int\d+)\s+',
                    r'^\s*(struct|union|enum)\s+\w+\s+',
                    r'^\s*\w+\s+\*?\w+\s*[;=]',  # 简单变量声明
                ]
                
                is_var_decl = False
                for pattern in var_patterns:
                    if re.match(pattern, stripped):
                        # 但如果包含函数调用，保留
                        if '(' in stripped and ')' in stripped:
                            break
                        is_var_decl = True
                        break
                
                if is_var_decl:
                    continue
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def get_string_at_address(self, ea):
        """尝试从地址获取字符串（增强版，支持更多编码）"""
        try:
            # 方法1：使用IDA的字符串识别
            string_type = idc.get_str_type(ea)
            if string_type is not None:
                content = idc.get_strlit_contents(ea, -1, string_type)
                if content:
                    decoded = content.decode('utf-8', errors='ignore') if isinstance(content, bytes) else str(content)
                    if self.is_meaningful_string(decoded):
                        return decoded
            
            # 方法2：手动读取ASCII字符串
            string_chars = []
            for i in range(512):  # 最大长度限制
                try:
                    byte_val = idc.get_wide_byte(ea + i)
                    if byte_val == 0:
                        break
                    if 32 <= byte_val <= 126:  # 可打印ASCII
                        string_chars.append(chr(byte_val))
                    else:
                        # 如果遇到非ASCII，尝试作为UTF-8
                        if byte_val > 127:
                            break
                        else:
                            break
                except:
                    break
            
            if len(string_chars) >= self.min_string_length:
                result = ''.join(string_chars)
                # 过滤无意义的字符串
                if self.is_meaningful_string(result):
                    return result
            
            # 方法3：尝试读取宽字符串（UTF-16）
            wide_string = self.try_read_wide_string(ea)
            if wide_string:
                return wide_string
                
        except Exception as e:
            logger.debug(f"Failed to get string at {ea:X}: {e}")
        return None
    
    def try_read_wide_string(self, ea):
        """尝试读取宽字符串（UTF-16）"""
        try:
            wide_chars = []
            for i in range(0, 256, 2):  # 宽字符是2字节
                try:
                    low_byte = idc.get_wide_byte(ea + i)
                    high_byte = idc.get_wide_byte(ea + i + 1)
                    
                    if low_byte == 0 and high_byte == 0:
                        break
                    
                    if high_byte == 0 and 32 <= low_byte <= 126:
                        wide_chars.append(chr(low_byte))
                    else:
                        break
                except:
                    break
            
            if len(wide_chars) >= self.min_string_length:
                result = ''.join(wide_chars)
                if self.is_meaningful_string(result):
                    return result
        except:
            pass
        
        return None
    
    def is_meaningful_string(self, s):
        """判断字符串是否有意义（增强版）"""
        if not s or len(s) < self.min_string_length:
            return False
        
        # 过滤太短或太长的字符串
        if len(s) > 300:
            return False
            
        # 过滤只包含单一字符的字符串
        if len(set(s)) <= 1:
            return False
        
        # 过滤常见的系统字符串和格式字符串
        meaningless_patterns = [
            r'^[%@\?\._\-\s]+$',     # 只包含格式字符
            r'^[0-9\s]+$',           # 只包含数字
            r'^[A-Z_]{1,3}$',        # 短的大写缩写
            r'^\s+$',                # 空白字符
            r'^[xX0]+$',             # 只有x或0
            r'^\.+$',                # 只有点
            r'^_+$',                 # 只有下划线
        ]
        
        for pattern in meaningless_patterns:
            if re.match(pattern, s):
                return False
        
        # 检查是否包含足够的字母字符
        alpha_count = sum(c.isalpha() for c in s)
        if alpha_count < 2:  # 至少2个字母
            return False
        
        # 过滤常见的无意义系统字符串
        meaningless_strings = [
            'NULL', 'null', 'nullptr',
            'true', 'false', 'TRUE', 'FALSE',
            'yes', 'no', 'YES', 'NO',
        ]
        
        if s in meaningless_strings:
            return False
        
        return True
    
    def extract_strings_from_function(self, func_ea):
        """从函数中提取字符串常量"""
        func = idaapi.get_func(func_ea)
        if not func:
            return []
        
        strings = set()  # 使用set避免重复
        current_ea = func.start_ea
        
        while current_ea < func.end_ea:
            try:
                insn = idaapi.insn_t()
                if idaapi.decode_insn(insn, current_ea):
                    # 检查所有操作数
                    for i in range(6):  # IDA最多6个操作数
                        op = insn.ops[i]
                        if op.type == idaapi.o_void:
                            break
                            
                        target_ea = None
                        if op.type == idaapi.o_imm:
                            target_ea = op.value
                        elif op.type == idaapi.o_mem:
                            target_ea = op.addr
                        elif op.type == idaapi.o_displ:
                            target_ea = op.addr
                        
                        if target_ea:
                            string_val = self.get_string_at_address(target_ea)
                            if string_val:
                                strings.add(string_val)
                
                current_ea = idc.next_head(current_ea, func.end_ea)
            except:
                current_ea = idc.next_head(current_ea, func.end_ea)
        
        return list(strings)
    
    def get_caller_context(self, func_ea):
        """获取函数的所有调用上下文（增强版）"""
        contexts = []
        
        # 获取所有对此函数的代码引用
        refs = list(idautils.CodeRefsTo(func_ea, 0))
        logger.debug(f"Found {len(refs)} references to function {self.get_function_name(func_ea)}")
        
        # 限制调用者数量，避免过多
        if len(refs) > self.max_callers:
            refs = refs[:self.max_callers]
            logger.debug(f"Limited to {self.max_callers} callers")
        
        for ref in refs:
            try:
                # 获取调用者函数
                caller_func_ea = idc.get_func_attr(ref, idc.FUNCATTR_START)
                if caller_func_ea == idaapi.BADADDR:
                    continue
                
                # Check symbol table first for resolved name
                caller_name = self.get_resolved_name(caller_func_ea)
                if not caller_name:
                    caller_name = self.get_function_name(caller_func_ea)
                
                # 过滤掉一些明显无用的调用者
                if self.is_dummy_function(caller_name):
                    continue
                
                # 获取调用点的汇编上下文
                asm_context = self.get_context_asm(ref, self.context_lines)
                
                # 尝试获取反编译上下文（改进版）
                decompiled_context = self.get_context_decompiled_enhanced(ref, caller_func_ea)
                
                # 分析调用参数中的字符串（增强版）
                call_strings = self.analyze_call_arguments_enhanced(ref, caller_func_ea)
                
                # 分析调用参数中的常量
                call_constants = self.analyze_call_constants(ref)
                
                context_info = {
                    'caller_ea': hex(caller_func_ea),
                    'caller_function': caller_name,
                    'call_site_ea': hex(ref),
                    'context_asm': asm_context,
                    'context_decompiled': decompiled_context,
                    'string_arguments': call_strings,
                    'constant_arguments': call_constants
                }
                
                contexts.append(context_info)
                
            except Exception as e:
                logger.debug(f"Failed to analyze caller context for ref {ref:X}: {e}")
        
        return contexts
    
    def is_dummy_function(self, func_name):
        """判断是否是无意义的哑函数"""
        if not func_name:
            return True
        
        # 常见的哑函数模式
        dummy_patterns = [
            r'^sub_[0-9A-F]+$',  # 仅有地址的函数
            r'^nullsub_',        # 空函数
            r'^j_j_',            # 多层跳转
            r'^_?_?thunk_',      # thunk函数
        ]
        
        for pattern in dummy_patterns:
            if re.match(pattern, func_name):
                return True
        
        return False
    
    def get_context_asm(self, ea, num_lines):
        """获取地址周围的汇编代码上下文"""
        context = []
        
        try:
            # 获取前面的指令
            prev_eas = []
            current = ea
            for _ in range(num_lines):
                prev = idc.prev_head(current)
                if prev == idaapi.BADADDR or prev >= current:
                    break
                prev_eas.insert(0, prev)
                current = prev
            
            # 添加前面的指令
            for prev_ea in prev_eas:
                disasm = idc.GetDisasm(prev_ea)
                if disasm:
                    context.append(f"{prev_ea:08X}: {disasm}")
            
            # 添加当前指令（标记为调用点）
            current_disasm = idc.GetDisasm(ea)
            if current_disasm:
                context.append(f"{ea:08X}: >>> {current_disasm} <<<")
            
            # 添加后面的指令
            current = ea
            for _ in range(num_lines):
                next_ea = idc.next_head(current)
                if next_ea == idaapi.BADADDR or next_ea <= current:
                    break
                disasm = idc.GetDisasm(next_ea)
                if disasm:
                    context.append(f"{next_ea:08X}: {disasm}")
                current = next_ea
                
        except Exception as e:
            logger.debug(f"Failed to get asm context for {ea:X}: {e}")
        
        return context
    
    def get_context_decompiled(self, call_ea, func_ea):
        """获取调用点的反编译上下文（改进版）"""
        try:
            if not ida_hexrays.init_hexrays_plugin():
                return []
            
            cfunc = ida_hexrays.decompile(func_ea)
            if not cfunc:
                return []
            
            # 获取反编译文本
            decompiled_text = str(cfunc)
            if not decompiled_text:
                return []
                
            lines = decompiled_text.split('\n')
            
            # 尝试找到包含调用的行
            target_func_name = self.get_function_name(idc.get_func_attr(call_ea, idc.FUNCATTR_START))
            called_func_ea = None
            
            # 获取被调用函数地址
            insn = idaapi.insn_t()
            if idaapi.decode_insn(insn, call_ea):
                if insn.Op1.type == idaapi.o_near:
                    called_func_ea = insn.Op1.addr
            
            called_func_name = self.get_function_name(called_func_ea) if called_func_ea else None
            
            # 查找调用位置
            call_line_idx = -1
            for i, line in enumerate(lines):
                if called_func_name and called_func_name in line:
                    call_line_idx = i
                    break
            
            # 提取上下文
            context_lines = []
            if call_line_idx >= 0:
                start = max(0, call_line_idx - self.context_lines)
                end = min(len(lines), call_line_idx + self.context_lines + 1)
                
                for i in range(start, end):
                    line = lines[i].strip()
                    if line:
                        if i == call_line_idx:
                            context_lines.append(f">>> {line} <<<")
                        else:
                            context_lines.append(line)
            else:
                # 如果找不到具体位置，返回函数开始部分
                for i, line in enumerate(lines[:self.context_lines * 2]):
                    line = line.strip()
                    if line and not line.startswith('//'):
                        context_lines.append(line)
            
            return context_lines
            
        except Exception as e:
            logger.debug(f"Failed to get decompiled context for {call_ea:X}: {e}")
            return []
    
    def get_context_decompiled_enhanced(self, call_ea, func_ea):
        """增强版反编译上下文提取，更精确定位调用点"""
        return self.get_context_decompiled(call_ea, func_ea)
    
    def analyze_call_arguments(self, call_ea):
        """分析函数调用的参数，提取字符串常量（增强版）"""
        string_args = []
        seen_strings = set()
        
        try:
            # 获取架构信息
            info = idaapi.get_inf_structure()
            is_64bit = info.is_64bit()
            
            # 向前搜索参数设置指令
            current_ea = call_ea
            search_range = self.arg_search_range
            
            for _ in range(search_range):
                prev_ea = idc.prev_head(current_ea)
                if prev_ea == idaapi.BADADDR or prev_ea >= current_ea:
                    break
                
                insn = idaapi.insn_t()
                if idaapi.decode_insn(insn, prev_ea):
                    # 检查更多可能的参数传递指令
                    param_setting_itypes = [
                        idaapi.NN_mov, idaapi.NN_lea, idaapi.NN_push,
                        idaapi.NN_movabs, idaapi.NN_movq, idaapi.NN_movd,
                        idaapi.NN_ldr, idaapi.NN_adr, idaapi.NN_adrp  # ARM指令
                    ]
                    
                    if insn.itype in param_setting_itypes:
                        for i in range(8):  # 检查更多操作数
                            op = insn.ops[i]
                            if op.type == idaapi.o_void:
                                break
                                
                            target_ea = None
                            if op.type == idaapi.o_imm:
                                target_ea = op.value
                            elif op.type == idaapi.o_mem:
                                target_ea = op.addr
                            elif op.type == idaapi.o_displ:
                                target_ea = op.addr
                            elif op.type == idaapi.o_phrase:
                                # 寄存器间接寻址，尝试获取地址
                                if op.addr:
                                    target_ea = op.addr
                            
                            if target_ea and target_ea not in seen_strings:
                                # 验证地址有效性
                                if not self.is_valid_address(target_ea):
                                    continue
                                
                                string_val = self.get_string_at_address(target_ea)
                                if string_val and len(string_val) >= self.min_string_length:
                                    seen_strings.add(target_ea)
                                    string_args.append({
                                        'value': string_val,
                                        'address': hex(target_ea),
                                        'instruction_ea': hex(prev_ea),
                                        'instruction': idc.GetDisasm(prev_ea)
                                    })
                
                current_ea = prev_ea
                
        except Exception as e:
            logger.debug(f"Failed to analyze call arguments for {call_ea:X}: {e}")
        
        return string_args
    
    def analyze_call_arguments_enhanced(self, call_ea, caller_func_ea):
        """增强版参数分析，结合反编译信息"""
        # 首先使用汇编层面的分析
        string_args = self.analyze_call_arguments(call_ea)
        
        # 尝试从反编译代码中提取更多字符串
        try:
            if ida_hexrays.init_hexrays_plugin():
                cfunc = ida_hexrays.decompile(caller_func_ea)
                if cfunc:
                    # 从反编译代码中搜索字符串字面量
                    decompiled_text = str(cfunc)
                    # 提取引号中的字符串
                    string_literals = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', decompiled_text)
                    
                    for lit in string_literals:
                        if len(lit) >= self.min_string_length and self.is_meaningful_string(lit):
                            # 检查是否已存在
                            if not any(arg['value'] == lit for arg in string_args):
                                string_args.append({
                                    'value': lit,
                                    'address': 'from_decompiled',
                                    'instruction_ea': hex(call_ea),
                                    'instruction': 'decompiled_literal'
                                })
        except Exception as e:
            logger.debug(f"Failed to extract strings from decompiled code: {e}")
        
        return string_args
    
    def analyze_call_constants(self, call_ea):
        """分析调用时使用的数值常量"""
        constants = []
        
        try:
            current_ea = call_ea
            search_range = min(15, self.arg_search_range)
            
            for _ in range(search_range):
                prev_ea = idc.prev_head(current_ea)
                if prev_ea == idaapi.BADADDR or prev_ea >= current_ea:
                    break
                
                insn = idaapi.insn_t()
                if idaapi.decode_insn(insn, prev_ea):
                    for i in range(6):
                        op = insn.ops[i]
                        if op.type == idaapi.o_void:
                            break
                        
                        # 收集立即数常量
                        if op.type == idaapi.o_imm:
                            value = op.value
                            # 过滤明显的地址
                            if 0 < value < 0x10000 or (value > 0x7FFFFFFF and value < 0xFFFFFFFF):
                                constants.append({
                                    'value': value,
                                    'hex': hex(value),
                                    'instruction_ea': hex(prev_ea)
                                })
                
                current_ea = prev_ea
        
        except Exception as e:
            logger.debug(f"Failed to analyze constants for {call_ea:X}: {e}")
        
        return constants[:10]  # 限制数量
    
    def is_valid_address(self, ea):
        """检查地址是否有效"""
        try:
            # 检查地址是否在有效段内
            seg = idaapi.getseg(ea)
            if not seg:
                return False
            
            # 检查是否可读
            if not (seg.perm & idaapi.SEGPERM_READ):
                return False
            
            return True
        except:
            return False
    
    def get_call_chain(self, func_ea, max_depth=None):
        """获取函数的调用链（增强版，过滤哑函数链）"""
        if max_depth is None:
            max_depth = self.max_call_depth
        
        def build_chains(ea, depth, visited, path):
            if depth >= max_depth or ea in visited:
                return []
            
            visited = visited.copy()
            visited.add(ea)
            chains = []
            
            try:
                refs = list(idautils.CodeRefsTo(ea, 0))
                if not refs:
                    # 到达调用链顶端
                    return [path] if path else [[]]
                
                valid_callers = 0
                for ref in refs:
                    caller_func_ea = idc.get_func_attr(ref, idc.FUNCATTR_START)
                    if caller_func_ea == idaapi.BADADDR or caller_func_ea in visited:
                        continue
                    
                    caller_name = self.get_function_name(caller_func_ea)
                    
                    # 过滤哑函数
                    if self.is_dummy_function(caller_name):
                        continue
                    
                    # 构建当前路径
                    current_path = [caller_name] + path
                    
                    # 递归构建上层调用链
                    upper_chains = build_chains(caller_func_ea, depth + 1, visited, current_path)
                    
                    if upper_chains:
                        chains.extend(upper_chains)
                        valid_callers += 1
                    else:
                        chains.append(current_path)
                        valid_callers += 1
                    
                    # 限制每层的分支数量
                    if valid_callers >= 5:
                        break
                
                # 如果没有有效调用者，返回当前路径
                if not chains and path:
                    return [path]
                        
            except Exception as e:
                logger.debug(f"Failed to build call chain for {ea:X}: {e}")
            
            return chains[:10]  # 限制返回的链数量
        
        try:
            func_name = self.get_function_name(func_ea)
            chains = build_chains(func_ea, 0, set(), [func_name])
            
            # 过滤和排序调用链
            filtered_chains = []
            for chain in chains:
                # 过滤只包含一个函数的链
                if len(chain) > 1:
                    # 计算调用链质量分数
                    score = self.calculate_chain_quality(chain)
                    filtered_chains.append((chain, score))
            
            # 按质量分数排序
            filtered_chains.sort(key=lambda x: x[1], reverse=True)
            
            # 返回前N条高质量调用链
            result_chains = [chain for chain, score in filtered_chains[:8]]
            
            return result_chains if result_chains else chains[:5]
        except Exception as e:
            logger.error(f"Error in get_call_chain: {e}")
            return []
    
    def calculate_chain_quality(self, chain):
        """计算调用链质量分数，用于排序"""
        score = 0
        
        # 长度适中的链更有价值
        length = len(chain)
        if 2 <= length <= 4:
            score += 10
        elif length == 5:
            score += 5
        
        # 检查函数名的语义价值
        for func_name in chain:
            # 有具体语义的函数名加分
            if not func_name.startswith('sub_'):
                score += 5
            
            # 常见的有意义的函数名模式
            meaningful_patterns = [
                r'main', r'init', r'create', r'process', r'handle',
                r'get', r'set', r'read', r'write', r'open', r'close',
                r'send', r'recv', r'parse', r'format', r'check'
            ]
            
            for pattern in meaningful_patterns:
                if re.search(pattern, func_name, re.IGNORECASE):
                    score += 3
                    break
        
        return score
    
    def analyze_function(self, func_ea):
        """分析单个函数并提取所有信息（优化版）"""
        try:
            func = idaapi.get_func(func_ea)
            if not func:
                return None
            
            # Check symbol table for current function name
            resolved_name = self.get_resolved_name(func_ea)
            func_name = resolved_name if resolved_name else self.get_function_name(func_ea)
            
            func_size = func.end_ea - func.start_ea
            
            if func_size < self.min_func_size:
                return None
            
            logger.debug(f"Analyzing function {func_name} at {func_ea:X}")
            
            # 1. 反编译函数（已优化，去除局部变量声明）
            function_body = self.decompile_function(func_ea)
            
            # 2. 获取内部字符串
            internal_strings = self.extract_strings_from_function(func_ea)
            
            # 3. 获取调用上下文（增强版，包含更准确的反编译上下文）
            caller_context = self.get_caller_context(func_ea)
            
            # 4. 获取调用链（增强版，过滤哑函数链）
            call_chain = self.get_call_chain(func_ea)
            
            # 5. 从调用上下文中收集字符串参数（去重）
            string_arguments = []
            seen_strings = set()
            for context in caller_context:
                for arg in context.get('string_arguments', []):
                    str_val = arg.get('value', '')
                    if str_val and str_val not in seen_strings:
                        seen_strings.add(str_val)
                        string_arguments.append(arg)
            
            # 6. 提取所有常量参数
            constant_arguments = []
            seen_constants = set()
            for context in caller_context:
                for const in context.get('constant_arguments', []):
                    const_val = const.get('value')
                    if const_val is not None and const_val not in seen_constants:
                        seen_constants.add(const_val)
                        constant_arguments.append(const)
            
            # 7. 统计调用者信息
            caller_functions = list(set([ctx['caller_function'] for ctx in caller_context]))
            
            # Initial population of symbol table (optional, based on existing symbols)
            if not func_name.startswith('sub_'):
                 self.update_symbol_table(func_ea, func_name)

            return {
                'func_ea': hex(func_ea),
                'func_end': hex(func.end_ea),
                'func_size': func_size,
                'dummy_name': func_name, # Note: this might be a resolved name now
                'real_name': 'None',
                'function_body': function_body,
                'caller_context': caller_context,
                'call_chain': call_chain,
                'string_arguments': string_arguments,
                'constant_arguments': constant_arguments,
                'internal_strings': internal_strings,
                'caller_count': len(caller_context),
                'caller_functions': caller_functions,
                'has_decompiled': 'return' in function_body or 'if' in function_body
            }
            
        except Exception as e:
            logger.error(f"Error analyzing function at {func_ea:X}: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return None
    
    def analyze_all_functions(self, output_path):
        """分析所有函数并导出到JSONL文件"""
        logger.info("Starting super enhanced function analysis...")
        
        self.wait_for_analysis()
        
        functions_analyzed = 0
        total_functions = len(list(idautils.Functions()))
        
        logger.info(f"Found {total_functions} functions to analyze")
        
        with jsonlines.open(output_path, mode='w') as writer:
            for func_ea in idautils.Functions():
                try:
                    result = self.analyze_function(func_ea)
                    if result:
                        writer.write(result)
                        functions_analyzed += 1
                        
                        if functions_analyzed % 50 == 0:
                            logger.info(f"Analyzed {functions_analyzed}/{total_functions} functions...")
                            
                except Exception as e:
                    logger.error(f"Error processing function at {func_ea:X}: {e}")
        
        logger.info(f"Analysis complete. Successfully analyzed {functions_analyzed} functions.")
        return functions_analyzed > 0

def main():
    """主函数"""
    logger.info("Super Enhanced IDA decompile script started")
    
    # 获取输出路径
    output_path = None
    if idc.ARGV and len(idc.ARGV) > 1:
        output_path = idc.ARGV[1]
    
    if not output_path:
        logger.error("No output path provided")
        idc.qexit(1)
        return
    
    logger.info(f"Output path: {output_path}")
    
    try:
        analyzer = SuperEnhancedFunctionAnalyzer()
        success = analyzer.analyze_all_functions(output_path)
        
        if success:
            logger.info("Script completed successfully")
            idc.qexit(0)
        else:
            logger.error("Script failed")
            idc.qexit(1)
            
    except Exception as e:
        logger.error(f"Script error: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # 创建错误输出文件
        try:
            with open(output_path, 'w') as f:
                json.dump({"error": str(e), "message": "Super enhanced analysis failed"}, f)
        except:
            pass
        
        idc.qexit(1)

if __name__ == "__main__":
    main()
