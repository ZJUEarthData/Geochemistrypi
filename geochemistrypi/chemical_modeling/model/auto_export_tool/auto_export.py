# 这个脚本是一个通用的地球化学仪器数据导出自动化工具，
# 支持多语言界面、智能坐标捕获、文件搜索和命名策略配置等功能，
# 旨在帮助用户快速创建适用于不同软件的数据导出自动化流程。
# 主要功能包括：
# 1. 多语言支持：提供中文和英文界面，用户可以根据需要；
# 2. 智能坐标捕获：用户可以通过移动鼠标并确认的方式捕获坐标，
#    支持重试和验证，确保坐标准确；
# 3. 文件搜索器：根据用户指定的文件夹、扩展名和是否搜索子文件夹
#    来查找数据文件，并提供文件列表供用户选择；
# 4. 文件名生成器：根据用户选择的命名策略（保持原名、添加时间戳、
#    添加序号或自定义模板）来生成输出文件名，确保导出文件有清晰的命名规则；
# 5. 配置管理器：用户可以创建、保存、加载和删除软件配置档案，
#    方便管理不同软件的导出流程；
# 6. 导出历史记录：自动记录每次批量导出的详细信息，
#    包括处理的文件、成功和失败的统计数据，用户可以随时查看历史记录；
# 7. 错误处理和用户提示：在整个流程中提供清晰的提示信息和错误处理机制，
#    帮助用户顺利完成配置和导出任务。
# 通过这个工具，用户可以轻松地为各种地球化学仪器软件创建定制化的
# 数据导出自动化流程，大大提高工作效率和数据处理的规范性。

# This script is a versatile automated tool for exporting geochemical instrument data,
# featuring a multi-language interface, intelligent coordinate capture, file search,
# and naming strategy configuration. It aims to assist users in quickly creating
# automated data export processes suitable for various software.
# Main functions include:
# 1. Multi-language support: Provides Chinese and English interfaces,
#    allowing users to choose according to their needs;
# 2. Intelligent coordinate capture: Users can capture coordinates by moving the mouse
#    and confirming, with support for retries and verification to ensure accurate coordinates;
# 3. File searcher: Searches for data files based on user-specified folder, extensions,
#    and subfolder search options, providing a list for selection;
# 4. Filename generator: Generates output filenames based on selected naming strategy
#    (keep original, add timestamp, add sequence number, or custom template),
#    ensuring clear naming conventions for exported files;
# 5. Configuration Manager: Allows users to create, save, load, and delete software
#    configuration profiles for managing export processes across different software;
# 6. Export history: Automatically records detailed information for each batch export,
#    including processed files and success/failure statistics, viewable anytime;
# 7. Error handling and user prompts: Provides clear prompts and error handling
#    throughout the process to help users complete configuration and export tasks successfully.
# With this tool, users can easily create customized automated data export processes
# for various geochemical instrument software, greatly improving work efficiency
# and the standardization of data processing.
import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pyautogui
import pygetwindow as gw
import pyperclip


# ==================== 多语言支持 ====================
class LanguageManager:
    """多语言管理器"""

    def __init__(self):
        self.language = "zh"  # 默认中文
        self.translations = {"zh": self._chinese_translations(), "en": self._english_translations()}

    def _chinese_translations(self):
        return {
            # 菜单
            "menu_title": "🌍 地球化学仪器数据通用导出工具",
            "menu_version": "版本 1.0 - 支持多语言和智能重试",
            "menu_options": "🆕 创建新的软件配置,🚀 批量导出数据,📊 查看导出历史,⚙️  管理配置档案,🛠️  编辑现有配置,🌐 切换语言,❌ 退出",
            "menu_prompt": "请选择操作 (1,2,3...): ",
            # 通用
            "invalid_choice": "❌ 无效选择",
            "confirm": "确认",
            "cancel": "取消",
            "retry": "重试",
            "back": "返回",
            "next": "下一步",
            "save": "保存",
            "exit": "退出",
            "success": "✅ 成功",
            "error": "❌ 错误",
            "warning": "⚠️  警告",
            "yes": "是",
            "no": "否",
            "files": "个文件",
            # 路径选择
            "path_selection_title": "📂 路径选择方式",
            "path_selection_custom": "1. 自定义路径",
            "path_selection_demo": "2. 演示路径",
            "path_selection_prompt": "请选择 (1/2): ",
            "demo_source_path": "Geochemistrypi/geochemistrypi/chemical_modeling/data",
            "demo_output_path": "Geochemistrypi/geopi_output",
            # 文件处理
            "select_source_folder": "📂 请输入数据文件所在文件夹路径:",
            "select_source_folder_with_options": "📂 请选择数据文件所在文件夹路径:",
            "folder_not_exist": "文件夹不存在",
            "create_folder": "文件夹 '{folder}' 不存在，是否创建? (y/n): ",
            "folder_created": "✅ 已创建文件夹: {folder}",
            "not_folder": "❌ 输入的不是文件夹路径",
            "searching_files": "🔍 正在搜索文件...",
            "search_folder": "文件夹: {folder}",
            "search_formats": "格式: {formats}",
            "search_subfolders": "搜索子文件夹: {status}",
            "files_found": "✅ 找到 {count} 个文件",
            "show_file_list": "是否显示文件列表? (y/n, 默认n): ",
            "select_files": "📋 选择要处理的文件:",
            "process_all": "处理所有文件",
            "process_specific": "选择特定文件",
            "select_specific_prompt": "选择要处理的文件 (输入文件编号，多个用逗号分隔):",
            "files_selected": "已选择 {count} 个文件进行处理",
            "and_more": "... 还有 {count} 个",
            # 坐标配置
            "coordinate_config": "📌 坐标配置",
            "coordinate_prompt": "请将鼠标移动到目标位置，然后按Enter键...",
            "coordinate_captured": "坐标已记录: ({x}, {y})",
            "coordinate_confirm": "确认此坐标? (y/n): ",
            "coordinate_retry": "要重新捕获此坐标吗? (y/n): ",
            "coordinate_failed": "❌ 坐标捕获失败",
            "coordinate_move_prompt": "请在 {seconds} 秒内将鼠标移动到目标位置...",
            "coordinate_countdown": "倒计时: {seconds} 秒...",
            "coordinate_final_confirm": "请确认所有坐标是否正确，如需修改请选择编号，确认请按0: ",
            "coordinate_edit_prompt": "选择要修改的坐标 (输入编号): ",
            # 鼠标动作
            "mouse_action_title": "📌 鼠标动作配置",
            "mouse_actions": "左键单击,右键单击,双击,拖拽",
            "mouse_wait_time": "操作后等待时间(秒，默认0.5): ",
            "mouse_description": "动作描述 (可选): ",
            # 键盘动作
            "keyboard_action_title": "⌨️  键盘动作配置",
            "keyboard_actions": "热键组合 (如Ctrl+S),单个按键,输入文本",
            "hotkey_prompt": "输入热键组合 (如 'ctrl+s'): ",
            "key_prompt": "输入按键: ",
            "text_prompt": "输入要输入的文本: ",
            "hotkey": "热键",
            "key_press": "按键",
            "type_text": "输入",
            # 剪贴板动作
            "clipboard_action_title": "📋 剪贴板动作配置",
            "clipboard_actions": "复制文本到剪贴板,从剪贴板粘贴",
            "copy_prompt": "输入要复制的内容: ",
            "copy_to_clipboard": "复制到剪贴板",
            "paste_from_clipboard": "从剪贴板粘贴",
            # 配置向导
            "wizard_title": "🛠️  创建新的软件自动化配置",
            "software_name": "请输入软件名称 (如 MC-ICP-MS,LA-ICP-MS等): ",
            "software_version": "软件版本 (可选): ",
            "software_description": "软件描述 (可选): ",
            "window_title": "软件窗口标题 (用于自动激活窗口): ",
            "file_extensions": "设置支持的文件扩展名 (例如: .dat, .txt, .raw): ",
            "action_recording": "🎬 现在开始录制动作序列...",
            "action_instructions": "请按照您平时导出数据的完整流程操作一遍",
            "action_recording_start": "按Enter键开始录制...",
            # 动作类型选择
            "action_selection": "请选择操作类型 (1,2,3...): ",
            "action_types": "鼠标点击,键盘输入,复制/粘贴,等待,完成配置",
            "wait_time_prompt": "等待时间(秒): ",
            "action_sequence": "动作序列",
            "no_actions_recorded": "尚未录制任何动作",
            "no_actions_to_undo": "没有可撤销的动作",
            "removed_last_action": "已移除上一步操作",
            # 文件格式配置
            "format_config": "📁 配置文件处理选项",
            "select_formats": "📄 选择要处理的文件格式:",
            "common_formats": "常用格式:",
            "custom_format": "自定义格式",
            "custom_format_prompt": "输入自定义格式 (如 '.dat,.txt,.raw'): ",
            "search_subfolders_prompt": "是否搜索子文件夹? (y/n, 默认y): ",
            # 命名策略
            "naming_strategy": "🏷️  选择输出文件命名策略:",
            "naming_options": "保持原文件名 [original.dat → original.csv],添加时间戳 [original.dat → original_20231215_143022.csv],添加序号 [original.dat → original_0001.csv],自定义模板",
            "custom_template_prompt": "输入自定义模板 (可用变量: {original_name}, {timestamp}, {date}, {time}, {index}, {total}): ",
            # 输出配置
            "output_config": "💾 配置输出选项",
            "output_format": "💾 选择输出文件格式:",
            "output_folder_prompt": "请输入输出文件夹路径 (留空则使用源文件夹):",
            "output_folder_prompt_with_options": "💾 请选择输出文件夹路径 (留空则使用源文件夹):",
            "use_source_folder": "使用源文件夹作为输出文件夹? (y/n): ",
            "batch_delay": "⏱️  文件处理间隔时间(秒, 默认2): ",
            # 预览
            "preview_title": "📋 导出设置预览",
            "software_config": "📊 软件配置: {name}",
            "source_folder": "📂 源文件夹: {folder}",
            "file_formats": "📄 文件格式: {formats}",
            "search_subfolders_status": "🔍 搜索子文件夹: {status}",
            "file_count": "📝 找到文件数: {count}",
            "naming_strategy_status": "🏷️  命名策略: {strategy}",
            "output_folder_status": "💾 输出文件夹: {folder}",
            "output_format_status": "🔄 输出格式: {format}",
            "filename_examples": "📝 文件名示例:",
            "confirm_export": "是否开始导出? (y/n): ",
            # 执行
            "export_starting": "🚀 开始批量导出",
            "activating_window": "🔍 正在激活软件窗口...",
            "window_activated": "✅ 已激活窗口: {title}",
            "window_not_found": "⚠️  未找到标题包含 '{title}' 的窗口",
            "manual_activate_prompt": "请手动激活目标软件窗口，然后按Enter键继续...",
            "processing_file": "📁 处理文件 {current}/{total}",
            "input_file": "输入: {file}",
            "output_file": "输出: {file}",
            "waiting_next": "⏳ 等待 {delay} 秒后处理下一个文件...",
            "export_success": "✅ 成功导出: {file}",
            "export_failed": "❌ 导出失败: {file}",
            "export_completed": "🎉 批量导出完成!",
            "statistics": "📊 处理统计:",
            "total_files": "总计文件: {count}",
            "success_count": "成功: {count}",
            "failed_count": "失败: {count}",
            "success_rate": "成功率: {rate:.1f}%",
            "failed_files": "❌ 失败的文件:",
            "start_time": "⏱️  开始时间: {time}",
            "end_time": "结束时间: {time}",
            "save_report": "是否保存详细结果报告? (y/n): ",
            "report_saved": "✅ 报告已保存: {file}",
            "export_report": "批量导出报告",
            "software_profile": "软件配置",
            "processing_time": "处理时间",
            "successful_exports": "成功导出的文件",
            "failed_exports": "失败的文件",
            "processing": "开始处理文件...",
            # 窗口相关
            "window_title_not_set": "未设置窗口标题",
            # 历史记录
            "history_title": "📊 导出历史记录 (共{count}条)",
            "no_history": "📭 暂无导出历史记录",
            "job_id": "作业ID: {id}",
            "job_software": "软件: {name}",
            "job_source": "源文件夹: {folder}",
            "job_file_count": "文件数: {count}",
            "job_status": "状态: {status}",
            "job_start_time": "开始时间: {time}",
            # 配置管理
            "manage_profiles": "管理配置档案",
            "no_profiles": "📭 暂无配置档案",
            "profile_list": "📋 配置档案列表:",
            "profile_name": "名称: {name}",
            "profile_desc": "描述: {desc}",
            "profile_created": "创建: {time}",
            "profile_formats": "文件类型: {formats}",
            "manage_options": "操作:",
            "delete_profile": "删除配置",
            "select_delete": "输入配置编号: ",
            "confirm_delete": "确认删除 '{name}'? (y/n): ",
            "delete_success": "✅ 删除成功",
            "delete_failed": "❌ 删除失败",
            "select_profile": "选择要编辑的配置",
            # 编辑配置
            "edit_profile": "🛠️  编辑配置: {name}",
            "edit_options": "修改基本信息,重新录制动作序列,返回",
            "edit_basic_info": "✏️  编辑基本信息",
            "edit_software_name": "软件名称 [{current}]: ",
            "edit_description": "描述 [{current}]: ",
            "edit_window_title": "窗口标题 [{current}]: ",
            "update_success": "✅ 基本信息已更新",
            "update_failed": "❌ 更新失败",
            "rerun_actions": "🎬 开始重新录制动作序列...",
            "actions_updated": "✅ 动作序列已更新",
            # 语言选择
            "language_title": "🌐 选择语言 / Select Language",
            "language_options": "1. 中文 (Chinese),2. English (英语)",
            "language_prompt": "请选择语言 / Please select language (1-2): ",
            "language_changed": "✅ 语言已切换为 {language}",
            # 错误信息
            "file_not_found": "未找到匹配的文件",
            "search_error": "搜索文件时出错: {error}",
            "input_error": "输入格式错误",
            "activation_error": "激活窗口失败: {error}",
            "export_error": "导出过程中出错: {error}",
            "file_error": "处理文件时出错: {error}",
            "program_interrupted": "⚠️  程序被用户中断",
            "program_error": "❌ 程序运行错误: {error}",
        }

    def _english_translations(self):
        return {
            # Menu
            "menu_title": "🌍 Universal Geochemical Instrument Data Export Tool",
            "menu_version": "Version 1.0 - Multi-language & Smart Retry",
            "menu_options": "🆕 Create new software profile,🚀 Batch export data,📊 View export history,⚙️  Manage profiles,🛠️  Edit existing profile,🌐 Switch language,❌ Exit",
            "menu_prompt": "Please select an option (1,2,3...): ",
            # General
            "invalid_choice": "❌ Invalid choice",
            "confirm": "Confirm",
            "cancel": "Cancel",
            "retry": "Retry",
            "back": "Back",
            "next": "Next",
            "save": "Save",
            "exit": "Exit",
            "success": "✅ Success",
            "error": "❌ Error",
            "warning": "⚠️  Warning",
            "yes": "Yes",
            "no": "No",
            "files": "files",
            # Path selection
            "path_selection_title": "📂 Path Selection Method",
            "path_selection_custom": "1. Custom Path",
            "path_selection_demo": "2. Demo Path",
            "path_selection_prompt": "Please select (1/2): ",
            "demo_source_path": "Geochemistrypi/geochemistrypi/chemical_modeling/data",
            "demo_output_path": "Geochemistrypi/geopi_output",
            # File processing
            "select_source_folder": "📂 Please enter the source folder path: ",
            "select_source_folder_with_options": "📂 Please select the source folder path:",
            "folder_not_exist": "Folder does not exist",
            "create_folder": "Folder '{folder}' does not exist, create it? (y/n): ",
            "folder_created": "✅ Folder created: {folder}",
            "not_folder": "❌ Input is not a folder path",
            "searching_files": "🔍 Searching files...",
            "search_folder": "Folder: {folder}",
            "search_formats": "Formats: {formats}",
            "search_subfolders": "Search subfolders: {status}",
            "files_found": "✅ Found {count} files",
            "show_file_list": "Show file list? (y/n, default n): ",
            "select_files": "📋 Select files to process:",
            "process_all": "Process all files",
            "process_specific": "Select specific files",
            "select_specific_prompt": "Select files to process (enter file numbers, separate with commas):",
            "files_selected": "Selected {count} files for processing",
            "and_more": "... and {count} more",
            # Coordinate configuration
            "coordinate_config": "📌 Coordinate Configuration",
            "coordinate_prompt": "Please move mouse to target position, then press Enter...",
            "coordinate_captured": "Coordinate captured: ({x}, {y})",
            "coordinate_confirm": "Confirm this coordinate? (y/n): ",
            "coordinate_retry": "Retry capturing this coordinate? (y/n): ",
            "coordinate_failed": "❌ Coordinate capture failed",
            "coordinate_move_prompt": "Please move mouse to target position in {seconds} seconds...",
            "coordinate_countdown": "Countdown: {seconds} seconds...",
            "coordinate_final_confirm": "Please confirm all coordinates are correct. Enter number to modify, 0 to confirm: ",
            "coordinate_edit_prompt": "Select coordinate to modify (enter number): ",
            # Mouse actions
            "mouse_action_title": "📌 Mouse Action Configuration",
            "mouse_actions": "Left Click,Right Click,Double Click,Drag",
            "mouse_wait_time": "Wait time after action (seconds, default 0.5): ",
            "mouse_description": "Action description (optional): ",
            # Keyboard actions
            "keyboard_action_title": "⌨️  Keyboard Action Configuration",
            "keyboard_actions": "Hotkey combination (e.g., Ctrl+S),Single key,Type text",
            "hotkey_prompt": "Enter hotkey combination (e.g., 'ctrl+s'): ",
            "key_prompt": "Enter key: ",
            "text_prompt": "Enter text to type: ",
            "hotkey": "Hotkey",
            "key_press": "Key press",
            "type_text": "Type text",
            # Clipboard actions
            "clipboard_action_title": "📋 Clipboard Action Configuration",
            "clipboard_actions": "Copy text to clipboard,Paste from clipboard",
            "copy_prompt": "Enter content to copy: ",
            "copy_to_clipboard": "Copy to clipboard",
            "paste_from_clipboard": "Paste from clipboard",
            # Configuration wizard
            "wizard_title": "🛠️  Create New Software Automation Profile",
            "software_name": "Enter software name (such as MC-ICP-MS,LA-ICP-MS...): ",
            "software_version": "Software version (optional): ",
            "software_description": "Software description (optional): ",
            "window_title": "Software window title (for auto-activation): ",
            "file_extensions": "Set supported file extensions (e.g., .dat, .txt, .raw): ",
            "action_recording": "🎬 Now recording action sequence...",
            "action_instructions": "Please perform your normal export workflow once",
            "action_recording_start": "Press Enter to start recording...",
            # Action type selection
            "action_selection": "Select action type (1,2,3...): ",
            "action_types": "Mouse click,Keyboard input,Copy/Paste,Wait,Finish configuration",
            "wait_time_prompt": "Wait time (seconds): ",
            "action_sequence": "Action sequence",
            "no_actions_recorded": "No actions recorded yet",
            "no_actions_to_undo": "No actions to undo",
            "removed_last_action": "Removed last action",
            # File format configuration
            "format_config": "📁 Configure File Processing Options",
            "select_formats": "📄 Select file formats to process:",
            "common_formats": "Common formats:",
            "custom_format": "Custom format",
            "custom_format_prompt": "Enter custom formats (e.g., '.dat,.txt,.raw'): ",
            "search_subfolders_prompt": "Search subfolders? (y/n, default y): ",
            # Naming strategy
            "naming_strategy": "🏷️  Select output file naming strategy:",
            "naming_options": (
                "Keep original name [original.dat → original.csv], "
                "Add timestamp [original.dat → original_20231215_143022.csv], "
                "Add sequence number [original.dat → original_0001.csv], "
                "Custom template"
            ),
            "custom_template_prompt": ("Enter custom template (available variables: " "{original_name}, {timestamp}, {date}, {time}, " "{index}, {total}): "),
            # Output configuration
            "output_config": "💾 Configure Output Options",
            "output_format": "💾 Select output file format:",
            "output_folder_prompt": "Enter output folder path (empty to use source folder): ",
            "output_folder_prompt_with_options": "💾 Please select the output folder path (empty to use source folder):",
            "use_source_folder": "Use source folder as output folder? (y/n): ",
            "batch_delay": "⏱️  Delay between files (seconds, default 2): ",
            # Preview
            "preview_title": "📋 Export Settings Preview",
            "software_config": "📊 Software profile: {name}",
            "source_folder": "📂 Source folder: {folder}",
            "file_formats": "📄 File formats: {formats}",
            "search_subfolders_status": "🔍 Search subfolders: {status}",
            "file_count": "📝 Files found: {count}",
            "naming_strategy_status": "🏷️  Naming strategy: {strategy}",
            "output_folder_status": "💾 Output folder: {folder}",
            "output_format_status": "🔄 Output format: {format}",
            "filename_examples": "📝 Filename examples:",
            "confirm_export": "Start export? (y/n): ",
            # Execution
            "export_starting": "🚀 Starting batch export",
            "activating_window": "🔍 Activating software window...",
            "window_activated": "✅ Window activated: {title}",
            "window_not_found": "⚠️  No window found with title containing '{title}'",
            "manual_activate_prompt": "Please manually activate target software window, then press Enter...",
            "processing_file": "📁 Processing file {current}/{total}",
            "input_file": "Input: {file}",
            "output_file": "Output: {file}",
            "waiting_next": "⏳ Waiting {delay} seconds before next file...",
            "export_success": "✅ Successfully exported: {file}",
            "export_failed": "❌ Export failed: {file}",
            "export_completed": "🎉 Batch export completed!",
            "statistics": "📊 Processing statistics:",
            "total_files": "Total files: {count}",
            "success_count": "Successful: {count}",
            "failed_count": "Failed: {count}",
            "success_rate": "Success rate: {rate:.1f}%",
            "failed_files": "❌ Failed files:",
            "start_time": "⏱️  Start time: {time}",
            "end_time": "End time: {time}",
            "save_report": "Save detailed report? (y/n): ",
            "report_saved": "✅ Report saved: {file}",
            "export_report": "Batch Export Report",
            "software_profile": "Software Profile",
            "processing_time": "Processing Time",
            "successful_exports": "Successful Exports",
            "failed_exports": "Failed Exports",
            "processing": "Starting file processing...",
            # Window related
            "window_title_not_set": "Window title not set",
            # History
            "history_title": "📊 Export History (Total: {count})",
            "no_history": "📭 No export history",
            "job_id": "Job ID: {id}",
            "job_software": "Software: {name}",
            "job_source": "Source folder: {folder}",
            "job_file_count": "Files: {count}",
            "job_status": "Status: {status}",
            "job_start_time": "Start time: {time}",
            # Profile management
            "manage_profiles": "Manage Profiles",
            "no_profiles": "📭 No profiles available",
            "profile_list": "📋 Profile List:",
            "profile_name": "Name: {name}",
            "profile_desc": "Description: {desc}",
            "profile_created": "Created: {time}",
            "profile_formats": "File types: {formats}",
            "manage_options": "Operations:",
            "delete_profile": "Delete profile",
            "select_delete": "Enter profile number: ",
            "confirm_delete": "Confirm delete '{name}'? (y/n): ",
            "delete_success": "✅ Delete successful",
            "delete_failed": "❌ Delete failed",
            "select_profile": "Select profile to edit",
            # Edit profile
            "edit_profile": "🛠️  Edit Profile: {name}",
            "edit_options": "Edit basic info,Re-record action sequence,Back",
            "edit_basic_info": "✏️  Edit Basic Information",
            "edit_software_name": "Software name [{current}]: ",
            "edit_description": "Description [{current}]: ",
            "edit_window_title": "Window title [{current}]: ",
            "update_success": "✅ Basic info updated",
            "update_failed": "❌ Update failed",
            "rerun_actions": "🎬 Starting to re-record action sequence...",
            "actions_updated": "✅ Action sequence updated",
            # Language selection
            "language_title": "🌐 Select Language / 选择语言",
            "language_options": "1. 中文 (Chinese),2. English (英语)",
            "language_prompt": "请选择语言 / Please select language (1-2): ",
            "language_changed": "✅ Language switched to {language}",
            # Error messages
            "file_not_found": "No matching files found",
            "search_error": "Error searching files: {error}",
            "input_error": "Input format error",
            "activation_error": "Failed to activate window: {error}",
            "export_error": "Error during export: {error}",
            "file_error": "Error processing file: {error}",
            "program_interrupted": "⚠️  Program interrupted by user",
            "program_error": "❌ Program runtime error: {error}",
        }

    def set_language(self, language):
        """设置语言"""
        if language in self.translations:
            self.language = language
            return True
        return False

    def get(self, key, **kwargs):
        """获取翻译文本"""
        text = self.translations[self.language].get(key, key)
        if kwargs:
            try:
                text = text.format(**kwargs)
            except Exception as e:
                print(f"错误: {e}")
        return text

    def t(self, key, **kwargs):
        """获取翻译的简写方法"""
        return self.get(key, **kwargs)


# ==================== 数据结构定义 ====================
@dataclass
class SoftwareProfile:
    """软件配置档案"""

    software_name: str
    version: str = ""
    description: str = ""
    window_title: str = ""
    actions: List[Dict[str, Any]] = field(default_factory=list)
    file_extensions: List[str] = field(default_factory=list)
    created_time: str = ""
    modified_time: str = ""

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


@dataclass
class FileProcessingConfig:
    """文件处理配置"""

    source_extensions: List[str] = field(default_factory=lambda: [".dat", ".txt", ".raw", ".csv"])
    search_subfolders: bool = True
    naming_strategy: str = "original"
    output_extension: str = ".csv"
    custom_naming_pattern: str = "{original_name}_exported_{timestamp}"
    output_folder: str = ""
    batch_delay: float = 2.0

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


@dataclass
class BatchJob:
    """批量作业"""

    profile_name: str
    source_folder: str
    file_config: FileProcessingConfig
    files_to_process: List[str] = field(default_factory=list)
    status: str = "pending"
    start_time: str = ""
    end_time: str = ""
    processed_count: int = 0
    failed_count: int = 0

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


# ==================== 交互式坐标捕获器 ====================
class CoordinateCapture:
    """智能坐标捕获器，支持重试和验证"""

    def __init__(self, lang_manager):
        self.lang = lang_manager

    def capture_with_countdown(self, description, countdown_seconds=5):
        """带倒计时的坐标捕获"""
        print(f"\n{self.lang.t('coordinate_config')}")
        print(f"{description}")

        while True:
            print(self.lang.t("coordinate_move_prompt", seconds=countdown_seconds))

            # 倒计时
            for i in range(countdown_seconds, 0, -1):
                print(self.lang.t("coordinate_countdown", seconds=i), end="\r")
                time.sleep(1)

            # 捕获坐标
            try:
                x, y = pyautogui.position()
                print(f"\n{self.lang.t('coordinate_captured', x=x, y=y)}")

                # 让用户确认
                confirm = input(f"{self.lang.t('coordinate_confirm')} ").lower()
                if confirm in ["y", "yes", "是", "确认"]:
                    return (x, y)
                else:
                    retry = input(f"{self.lang.t('coordinate_retry')} ").lower()
                    if retry not in ["y", "yes", "是", "确认"]:
                        return None
            except Exception as e:
                print(f"{self.lang.t('coordinate_failed')}: {str(e)}")
                retry = input(f"{self.lang.t('coordinate_retry')} ").lower()
                if retry not in ["y", "yes", "是", "确认"]:
                    return None

    def capture_instant(self, description):
        """即时坐标捕获"""
        print(f"\n{self.lang.t('coordinate_config')}")
        print(f"{description}")

        while True:
            input(f"{self.lang.t('coordinate_prompt')}")

            try:
                x, y = pyautogui.position()
                print(f"{self.lang.t('coordinate_captured', x=x, y=y)}")

                # 让用户确认
                confirm = input(f"{self.lang.t('coordinate_confirm')} ").lower()
                if confirm in ["y", "yes", "是", "确认"]:
                    return (x, y)
                else:
                    retry = input(f"{self.lang.t('coordinate_retry')} ").lower()
                    if retry not in ["y", "yes", "是", "确认"]:
                        return None
            except Exception as e:
                print(f"{self.lang.t('coordinate_failed')}: {str(e)}")
                retry = input(f"{self.lang.t('coordinate_retry')} ").lower()
                if retry not in ["y", "yes", "是", "确认"]:
                    return None


# ==================== 配置管理器 ====================
class ConfigManager:
    def __init__(self, config_dir=None):
        if config_dir is None:
            config_dir = os.path.join(os.path.dirname(__file__), "software_profiles")
        self.config_dir = config_dir
        self.history_file = os.path.join(config_dir, "export_history.json")
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)

    def list_profiles(self):
        """列出所有可用的软件配置"""
        profiles = []
        for file in os.listdir(self.config_dir):
            if file.endswith(".json") and file != "export_history.json":
                profiles.append(file[:-5])
        return profiles

    def save_profile(self, profile: SoftwareProfile):
        """保存软件配置"""
        profile.modified_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not profile.created_time:
            profile.created_time = profile.modified_time

        filename = f"{profile.software_name}.json"
        filepath = os.path.join(self.config_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(profile.to_dict(), f, indent=2, ensure_ascii=False)
        return True

    def load_profile(self, profile_name: str) -> Optional[SoftwareProfile]:
        """加载软件配置"""
        filepath = os.path.join(self.config_dir, f"{profile_name}.json")
        if not os.path.exists(filepath):
            return None

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return SoftwareProfile.from_dict(data)

    def delete_profile(self, profile_name: str):
        """删除软件配置"""
        filepath = os.path.join(self.config_dir, f"{profile_name}.json")
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
        return False

    def save_export_history(self, job: BatchJob):
        """保存导出历史记录"""
        history = self.load_export_history()
        job_dict = job.to_dict()
        job_dict["job_id"] = f"job_{int(time.time())}"
        history.append(job_dict)

        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    def load_export_history(self) -> List[Dict]:
        """加载导出历史记录"""
        if os.path.exists(self.history_file):
            with open(self.history_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return []


# ==================== 文件搜索器 ====================
class FileSearcher:
    @staticmethod
    def find_files(folder_path: str, extensions: List[str], search_subfolders: bool = True) -> List[str]:
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder does not exist: {folder_path}")

        matched_files = []

        if search_subfolders:
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    file_ext = os.path.splitext(file)[1].lower()
                    if file_ext in extensions:
                        full_path = os.path.join(root, file)
                        matched_files.append(full_path)
        else:
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if os.path.isfile(file_path):
                    file_ext = os.path.splitext(file)[1].lower()
                    if file_ext in extensions:
                        matched_files.append(file_path)

        matched_files.sort()
        return matched_files

    @staticmethod
    def group_files_by_extension(files: List[str]) -> Dict[str, List[str]]:
        groups = {}
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext not in groups:
                groups[ext] = []
            groups[ext].append(file)
        return groups


# ==================== 文件名生成器 ====================
class FilenameGenerator:
    def __init__(self, config: FileProcessingConfig):
        self.config = config
        self.counter = 1

    def generate_output_filename(self, source_filepath: str, index: int = None, total: int = None) -> str:
        source_filename = os.path.basename(source_filepath)
        source_name, source_ext = os.path.splitext(source_filename)
        source_folder = os.path.dirname(source_filepath)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.config.naming_strategy == "original":
            output_name = f"{source_name}{self.config.output_extension}"
        elif self.config.naming_strategy == "timestamp":
            output_name = f"{source_name}_{timestamp}{self.config.output_extension}"
        elif self.config.naming_strategy == "counter":
            if index is not None:
                counter_str = f"{index:04d}"
                if total is not None:
                    counter_str = f"{index:04d}_of_{total:04d}"
                output_name = f"{source_name}_{counter_str}{self.config.output_extension}"
            else:
                output_name = f"{source_name}_{self.counter:04d}{self.config.output_extension}"
                self.counter += 1
        elif self.config.naming_strategy == "custom":
            template = self.config.custom_naming_pattern
            output_name = template.format(
                original_name=source_name,
                original_ext=source_ext[1:],
                folder_name=os.path.basename(source_folder),
                timestamp=timestamp,
                date=datetime.now().strftime("%Y%m%d"),
                time=datetime.now().strftime("%H%M%S"),
                index=index if index else self.counter,
                total=total if total else "unknown",
            )
            output_name += self.config.output_extension
        else:
            output_name = f"{source_name}_exported{self.config.output_extension}"

        return output_name

    @staticmethod
    def get_unique_filename(output_folder: str, filename: str) -> str:
        base_name, ext = os.path.splitext(filename)
        counter = 1
        unique_name = filename

        while os.path.exists(os.path.join(output_folder, unique_name)):
            unique_name = f"{base_name}_{counter:03d}{ext}"
            counter += 1

        return unique_name


# ==================== 交互式配置向导（增强版） ====================
class EnhancedSetupWizard:
    """增强版配置向导，支持多语言和重试"""

    def __init__(self, config_manager: ConfigManager, lang_manager: LanguageManager):
        self.cm = config_manager
        self.lang = lang_manager
        self.coord_capturer = CoordinateCapture(lang_manager)

    def create_new_profile(self):
        """创建新的软件配置向导"""
        print("\n" + "=" * 60)
        print(self.lang.t("wizard_title"))
        print("=" * 60)

        # 基本信息配置
        profile = self._configure_basic_info()
        if not profile:
            return None

        # 录制动作序列
        actions = self._record_actions_interactive()
        if actions is None:  # 用户取消
            return None

        profile.actions = actions

        # 保存配置
        if self.cm.save_profile(profile):
            print(f"{self.lang.t('success')}: {self.lang.t('profile_created', name=profile.software_name)}")
            return profile
        else:
            print(self.lang.t("error"))
            return None

    def _configure_basic_info(self):
        """配置基本信息"""
        while True:
            try:
                software_name = input(self.lang.t("software_name"))
                if not software_name:
                    print(self.lang.t("warning"))
                    continue

                version = input(self.lang.t("software_version"))
                description = input(self.lang.t("software_description"))
                window_title = input(self.lang.t("window_title"))

                # 文件扩展名
                print(f"\n{self.lang.t('file_extensions')}")
                extensions = input(": ").replace(" ", "").split(",")
                file_extensions = [ext if ext.startswith(".") else f".{ext}" for ext in extensions if ext]

                # 确认信息
                print(f"\n{self.lang.t('preview_title')}")
                print(f"{self.lang.t('software_name')}: {software_name}")
                print(f"{self.lang.t('software_version')}: {version}")
                print(f"{self.lang.t('software_description')}: {description}")
                print(f"{self.lang.t('window_title')}: {window_title}")
                print(f"{self.lang.t('file_formats', formats=', '.join(file_extensions))}")

                confirm = input(f"\n{self.lang.t('confirm')}? (y/n): ").lower()
                if confirm in ["y", "yes", "是", "确认"]:
                    return SoftwareProfile(software_name=software_name, version=version, description=description, window_title=window_title, file_extensions=file_extensions)
                else:
                    retry = input(f"{self.lang.t('retry')}? (y/n): ").lower()
                    if retry not in ["y", "yes", "是", "确认"]:
                        return None
            except Exception as e:
                print(f"{self.lang.t('error')}: {str(e)}")
                continue

    def _record_actions_interactive(self):
        """交互式录制动作，支持重做每一步"""
        actions = []
        step_count = 0

        print(f"\n{self.lang.t('action_recording')}")
        print(self.lang.t("action_instructions"))

        input(f"{self.lang.t('action_recording_start')}")

        while True:
            step_count += 1
            print(f"\n{self.lang.t('action_selection')}")

            # 显示动作类型选项
            action_types = self.lang.t("action_types").split(",")
            for i, action_type in enumerate(action_types, 1):
                print(f"  {i}. {action_type.strip()}")

            print(f"  6. {self.lang.t('back')} (撤销上一步)")
            print(f"  7. {self.lang.t('save')} (完成录制)")

            choice = input(f"{self.lang.t('action_selection')}")

            if choice == "1":
                action = self._record_mouse_action()
                if action:
                    actions.append(action)
            elif choice == "2":
                action = self._record_keyboard_action()
                if action:
                    actions.append(action)
            elif choice == "3":
                action = self._record_clipboard_action()
                if action:
                    actions.append(action)
            elif choice == "4":
                wait_time = float(input(f"{self.lang.t('wait_time_prompt')}"))
                actions.append({"type": "wait", "wait_time": wait_time})
            elif choice == "5":
                # 查看当前动作序列
                self._review_actions(actions)
                continue
            elif choice == "6":
                # 撤销上一步
                if actions:
                    actions.pop()
                    print(f"{self.lang.t('success')}: {self.lang.t('removed_last_action')}")
                else:
                    print(self.lang.t("warning") + ": " + self.lang.t("no_actions_to_undo"))
                continue
            elif choice == "7":
                # 完成录制
                if self._confirm_actions(actions):
                    return actions
                else:
                    continue
            else:
                print(self.lang.t("invalid_choice"))

    def _record_mouse_action(self):
        """记录鼠标动作"""
        print(f"\n{self.lang.t('mouse_action_title')}")
        mouse_actions = self.lang.t("mouse_actions").split(",")
        for i, action in enumerate(mouse_actions, 1):
            print(f"  {i}. {action.strip()}")

        while True:
            choice = input(f"{self.lang.t('action_selection')}")
            action_type_map = {"1": "click", "2": "right_click", "3": "double_click", "4": "drag"}

            if choice not in action_type_map:
                print(self.lang.t("invalid_choice"))
                continue

            description = input(f"{self.lang.t('mouse_description')}")
            wait_time = float(input(f"{self.lang.t('mouse_wait_time')}") or "0.5")

            # 捕获坐标
            action_desc = mouse_actions[int(choice) - 1].strip()
            coord_desc = f"{action_desc}: {description}"
            coordinates = self.coord_capturer.capture_instant(coord_desc)

            if coordinates:
                return {"type": "mouse", "action": action_type_map[choice], "coordinates": list(coordinates), "wait_time": wait_time, "description": description}
            else:
                retry = input(f"{self.lang.t('retry')}? (y/n): ").lower()
                if retry not in ["y", "yes", "是", "确认"]:
                    return None

    def _record_keyboard_action(self):
        """记录键盘动作"""
        print(f"\n{self.lang.t('keyboard_action_title')}")
        keyboard_actions = self.lang.t("keyboard_actions").split(",")
        for i, action in enumerate(keyboard_actions, 1):
            print(f"  {i}. {action.strip()}")

        choice = input(f"{self.lang.t('action_selection')}")

        if choice == "1":
            keys = input(f"{self.lang.t('hotkey_prompt')}").split("+")
            return {"type": "keyboard", "action": "hotkey", "keys": keys}
        elif choice == "2":
            key = input(f"{self.lang.t('key_prompt')}")
            return {"type": "keyboard", "action": "press", "key": key}
        elif choice == "3":
            text = input(f"{self.lang.t('text_prompt')}")
            return {"type": "keyboard", "action": "type", "text": text}
        return None

    def _record_clipboard_action(self):
        """记录剪贴板动作"""
        print(f"\n{self.lang.t('clipboard_action_title')}")
        clipboard_actions = self.lang.t("clipboard_actions").split(",")
        for i, action in enumerate(clipboard_actions, 1):
            print(f"  {i}. {action.strip()}")

        choice = input(f"{self.lang.t('action_selection')}")

        if choice == "1":
            content = input(f"{self.lang.t('copy_prompt')}")
            return {"type": "clipboard", "action": "copy", "content": content}
        elif choice == "2":
            return {"type": "clipboard", "action": "paste"}
        return None

    def _review_actions(self, actions):
        """查看当前动作序列"""
        if not actions:
            print(self.lang.t("warning") + ": " + self.lang.t("no_actions_recorded"))
            return

        print(f"\n{self.lang.t('preview_title')} - {self.lang.t('action_sequence')}")
        print("-" * 60)
        for i, action in enumerate(actions, 1):
            action_type = action.get("type", "unknown")
            if action_type == "mouse":
                desc = action.get("description", "")
                coords = action.get("coordinates", [0, 0])
                print(f"{i}. 🖱️  {action.get('action', 'click')} ({coords[0]}, {coords[1]}) - {desc}")
            elif action_type == "keyboard":
                action_name = action.get("action", "")
                if action_name == "hotkey":
                    keys = "+".join(action.get("keys", []))
                    print(f"{i}. ⌨️  热键: {keys}")
                elif action_name == "press":
                    key = action.get("key", "")
                    print(f"{i}. ⌨️  按键: {key}")
                else:
                    text = action.get("text", "")[:50]
                    print(f"{i}. ⌨️  输入: {text}...")
            elif action_type == "clipboard":
                action_name = action.get("action", "")
                if action_name == "copy":
                    content = action.get("content", "")[:50]
                    print(f"{i}. 📋 复制: {content}...")
                else:
                    print(f"{i}. 📋 粘贴")
            elif action_type == "wait":
                wait_time = action.get("wait_time", 0)
                print(f"{i}. ⏱️  等待: {wait_time}秒")
        print("-" * 60)

    def _confirm_actions(self, actions):
        """确认动作序列"""
        if not actions:
            print(self.lang.t("warning") + ": " + self.lang.t("no_actions_recorded"))
            return False

        self._review_actions(actions)

        confirm = input(f"\n{self.lang.t('confirm')} {self.lang.t('action_sequence')}? (y/n): ").lower()
        return confirm in ["y", "yes", "是", "确认"]


# ==================== 自动化执行器（多语言版） ====================
class AutomationExecutor:
    def __init__(self, profile: SoftwareProfile, lang_manager: LanguageManager):
        self.profile = profile
        self.lang = lang_manager
        self.current_file_index = 0
        self.total_files = 0

    def activate_window(self):
        """激活目标软件窗口"""
        if not self.profile.window_title:
            print(f"{self.lang.t('warning')}: {self.lang.t('window_title_not_set')}")
            input(f"{self.lang.t('manual_activate_prompt')}")
            return True

        try:
            windows = gw.getWindowsWithTitle(self.profile.window_title)
            if windows:
                win = windows[0]
                win.activate()
                time.sleep(0.5)
                win.maximize()
                time.sleep(0.5)
                print(self.lang.t("window_activated", title=win.title))
                return True
            else:
                print(self.lang.t("window_not_found", title=self.profile.window_title))
                input(f"{self.lang.t('manual_activate_prompt')}")
                return True
        except Exception as e:
            print(f"{self.lang.t('activation_error', error=str(e))}")
            return False

    def execute_batch(self, batch_job: BatchJob) -> Dict[str, Any]:
        """
        执行批量导出作业

        Returns:
            执行结果统计
        """
        results = {"total": len(batch_job.files_to_process), "success": 0, "failed": 0, "failed_files": [], "output_files": []}

        # 初始化文件名生成器
        file_config = batch_job.file_config
        name_generator = FilenameGenerator(file_config)

        # 确保输出文件夹存在
        output_folder = file_config.output_folder
        if output_folder and not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)

        # 批量处理文件
        for idx, source_file in enumerate(batch_job.files_to_process, 1):
            print(f"\n{'='*60}")
            print(self.lang.t("processing_file", current=idx, total=len(batch_job.files_to_process)))
            print(self.lang.t("input_file", file=source_file))

            try:
                # 生成输出文件名
                output_filename = name_generator.generate_output_filename(source_file, index=idx, total=len(batch_job.files_to_process))

                # 确保文件名唯一
                if output_folder:
                    output_filename = FilenameGenerator.get_unique_filename(output_folder, output_filename)
                    output_path = os.path.join(output_folder, output_filename)
                else:
                    # 如果没有指定输出文件夹，使用源文件所在文件夹
                    source_dir = os.path.dirname(source_file)
                    output_filename = FilenameGenerator.get_unique_filename(source_dir, output_filename)
                    output_path = os.path.join(source_dir, output_filename)

                print(self.lang.t("output_file", file=output_filename))

                # 准备变量替换
                variables = {
                    "{INPUT_FILE}": source_file,
                    "{OUTPUT_FILE}": output_path,
                    "{INPUT_FILENAME}": os.path.basename(source_file),
                    "{INPUT_BASENAME}": os.path.splitext(os.path.basename(source_file))[0],
                    "{OUTPUT_FILENAME}": os.path.basename(output_path),
                    "{OUTPUT_BASENAME}": os.path.splitext(os.path.basename(output_path))[0],
                    "{FILE_INDEX}": str(idx),
                    "{TOTAL_FILES}": str(len(batch_job.files_to_process)),
                    "{TIMESTAMP}": datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "{DATE}": datetime.now().strftime("%Y%m%d"),
                    "{TIME}": datetime.now().strftime("%H%M%S"),
                }

                # 执行导出动作
                success = self._execute_single_export(source_file, output_path, variables)

                if success:
                    print(self.lang.t("export_success", file=output_filename))
                    results["success"] += 1
                    results["output_files"].append(output_path)
                else:
                    print(self.lang.t("export_failed", file=os.path.basename(source_file)))
                    results["failed"] += 1
                    results["failed_files"].append(source_file)

            except Exception as e:
                print(f"{self.lang.t('file_error', error=str(e))}")
                results["failed"] += 1
                results["failed_files"].append(source_file)

            # 文件间等待（避免软件响应不过来）
            if idx < len(batch_job.files_to_process):
                print(self.lang.t("waiting_next", delay=file_config.batch_delay))
                time.sleep(file_config.batch_delay)

        return results

    def _execute_single_export(self, input_file: str, output_file: str, variables: Dict[str, str]) -> bool:
        """执行单个文件的导出"""
        try:
            for i, action in enumerate(self.profile.actions, 1):
                action_type = action.get("type")

                if action_type == "mouse":
                    self._execute_mouse_action(action, variables)
                elif action_type == "keyboard":
                    self._execute_keyboard_action(action, variables)
                elif action_type == "clipboard":
                    self._execute_clipboard_action(action, variables)
                elif action_type == "wait":
                    wait_time = action.get("wait_time", 1)
                    time.sleep(wait_time)

                # 动作后等待
                time.sleep(action.get("wait_time", 0.3))

            return True

        except Exception as e:
            print(f"{self.lang.t('export_error', error=str(e))}")
            # 尝试按ESC键取消可能弹出的对话框
            for _ in range(3):
                pyautogui.press("esc")
                time.sleep(0.5)
            return False

    def _execute_mouse_action(self, action: Dict, variables: Dict):
        """执行鼠标动作"""
        mouse_action = action.get("action", "click")
        coords = action.get("coordinates", [0, 0])

        x, y = coords[0], coords[1]

        if mouse_action == "click":
            pyautogui.click(x, y)
        elif mouse_action == "right_click":
            pyautogui.rightClick(x, y)
        elif mouse_action == "double_click":
            pyautogui.doubleClick(x, y)
        elif mouse_action == "drag":
            pyautogui.dragTo(x, y, duration=0.5)

        desc = action.get("description", "")
        if desc:
            print(f"  🖱️  {desc} ({x}, {y})")

    def _execute_keyboard_action(self, action: Dict, variables: Dict):
        """执行键盘动作"""
        kb_action = action.get("action", "press")

        if kb_action == "hotkey":
            keys = action.get("keys", [])
            if keys:
                pyautogui.hotkey(*keys)
                print(f"  ⌨️  {self.lang.t('hotkey')}: {'+'.join(keys)}")
        elif kb_action == "press":
            key = action.get("key", "")
            if key:
                pyautogui.press(key)
                print(f"  ⌨️  {self.lang.t('key_press')}: {key}")
        elif kb_action == "type":
            text = action.get("text", "")
            # 替换变量
            for var_name, var_value in variables.items():
                text = text.replace(var_name, var_value)
            pyautogui.write(text)
            print(f"  ⌨️  {self.lang.t('type_text')}: {text[:50]}...")

    def _execute_clipboard_action(self, action: Dict, variables: Dict):
        """执行剪贴板动作"""
        action_type = action.get("action", "")
        content = action.get("content", "")

        if action_type == "copy":
            # 替换变量
            for var_name, var_value in variables.items():
                content = content.replace(var_name, var_value)
            pyperclip.copy(content)
            print(f"  📋 {self.lang.t('copy_to_clipboard')}: {content[:50]}...")
        elif action_type == "paste":
            pyautogui.hotkey("ctrl", "v")
            print(f"  📋 {self.lang.t('paste_from_clipboard')}")


# ==================== 主程序界面（多语言版） ====================
class MultilingualExportAssistant:
    def __init__(self):
        self.lang = LanguageManager()
        self.config_manager = ConfigManager()
        self.setup_wizard = EnhancedSetupWizard(self.config_manager, self.lang)

    def select_language(self):
        """选择语言"""
        print("\n" + "=" * 60)
        print(self.lang.t("language_title"))
        print("=" * 60)

        language_options = self.lang.t("language_options").split(",")
        for option in language_options:
            print(f"  {option.strip()}")

        while True:
            choice = input(f"\n{self.lang.t('language_prompt')}")
            if choice == "1":
                self.lang.set_language("zh")
                print(self.lang.t("language_changed", language="中文"))
                break
            elif choice == "2":
                self.lang.set_language("en")
                print(self.lang.t("language_changed", language="English"))
                break
            else:
                print(self.lang.t("invalid_choice"))

    def main_menu(self):
        """主菜单"""
        while True:
            print("\n" + "=" * 60)
            print(self.lang.t("menu_title"))
            print(self.lang.t("menu_version"))
            print("=" * 60)

            menu_options = self.lang.t("menu_options").split(",")
            for i, option in enumerate(menu_options, 1):
                print(f"{i}. {option.strip()}")

            choice = input(f"\n{self.lang.t('menu_prompt')}")

            if choice == "1":
                self.create_profile()
            elif choice == "2":
                self.batch_export()
            elif choice == "3":
                self.view_export_history()
            elif choice == "4":
                self.manage_profiles()
            elif choice == "5":
                self.edit_profile()
            elif choice == "6":
                self.select_language()
            elif choice == "7":
                print(f"\n{self.lang.t('exit')}...")
                break
            else:
                print(self.lang.t("invalid_choice"))

    def create_profile(self):
        """创建新的软件配置"""
        profile = self.setup_wizard.create_new_profile()
        if profile:
            print(f"{self.lang.t('success')}: {self.lang.t('profile_created', name=profile.software_name)}")

    def batch_export(self):
        """批量导出主流程"""
        # 选择软件配置
        profile = self._select_profile()
        if not profile:
            return

        print(f"\n{self.lang.t('software_config', name=profile.software_name)}")

        # 配置文件处理选项
        file_config = self._configure_file_processing()
        if not file_config:
            return

        # 选择源文件夹（带路径选择选项）
        source_folder = self._select_source_folder_with_options()
        if not source_folder:
            return

        # 搜索文件
        files = self._search_and_select_files(source_folder, file_config)
        if not files:
            return

        # 配置输出选项（带路径选择选项）
        output_folder = self._configure_output_options_with_options()
        file_config.output_folder = output_folder

        # 预览和确认
        if not self._preview_and_confirm(profile, source_folder, files, file_config):
            return

        # 创建批量作业并执行
        batch_job = BatchJob(
            profile_name=profile.software_name, source_folder=source_folder, file_config=file_config, files_to_process=files, status="pending", start_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        self._execute_batch_export(profile, batch_job)
        self.config_manager.save_export_history(batch_job)

    def _select_profile(self):
        """选择软件配置"""
        profiles = self.config_manager.list_profiles()
        if not profiles:
            print(f"\n{self.lang.t('no_profiles')}")
            return None

        print(f"\n{self.lang.t('profile_list')}")
        for i, profile_name in enumerate(profiles, 1):
            profile = self.config_manager.load_profile(profile_name)
            print(f"  {i}. {profile_name} - {profile.description}")

        while True:
            try:
                choice = input(f"\n{self.lang.t('select_delete')}")
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(profiles):
                    return self.config_manager.load_profile(profiles[choice_idx])
                else:
                    print(self.lang.t("invalid_choice"))
            except ValueError:
                print(self.lang.t("input_error"))

    def _configure_file_processing(self):
        """配置文件处理选项"""
        print(f"\n{self.lang.t('format_config')}")
        print("=" * 40)

        config = FileProcessingConfig()

        # 选择文件格式
        print(f"\n{self.lang.t('select_formats')}")
        common_formats = [".dat", ".txt", ".csv", ".raw", ".xlsx", ".xls", ".json", ".xml"]

        print(f"{self.lang.t('common_formats')}")
        for i, fmt in enumerate(common_formats, 1):
            print(f"  {i}. {fmt}")
        print(f"  0. {self.lang.t('custom_format')}")

        while True:
            try:
                format_choice = input(f"\n{self.lang.t('select_formats')}: ")

                if format_choice == "0":
                    custom_formats = input(f"{self.lang.t('custom_format_prompt')}: ")
                    config.source_extensions = [ext.strip() if ext.startswith(".") else f".{ext.strip()}" for ext in custom_formats.split(",")]
                    break
                else:
                    selected_indices = [int(idx.strip()) for idx in format_choice.split(",")]
                    selected_formats = [common_formats[idx - 1] for idx in selected_indices if 1 <= idx <= len(common_formats)]
                    if selected_formats:
                        config.source_extensions = selected_formats
                        break
                    else:
                        print(self.lang.t("input_error"))
            except ValueError:
                print(self.lang.t("input_error"))

        # 是否搜索子文件夹
        search_sub = input(f"\n{self.lang.t('search_subfolders_prompt')}").lower()
        config.search_subfolders = search_sub != "n"

        # 选择命名策略
        print(f"\n{self.lang.t('naming_strategy')}")
        naming_options = self.lang.t("naming_options").split(",")
        for i, option in enumerate(naming_options, 1):
            print(f"  {i}. {option.strip()}")

        naming_choice = input(f"\n{self.lang.t('action_selection')}") or "1"

        if naming_choice == "1":
            config.naming_strategy = "original"
        elif naming_choice == "2":
            config.naming_strategy = "timestamp"
        elif naming_choice == "3":
            config.naming_strategy = "counter"
        elif naming_choice == "4":
            config.naming_strategy = "custom"
            template = input(f"{self.lang.t('custom_template_prompt')}: ")
            config.custom_naming_pattern = template or "{original_name}_exported_{timestamp}"
        else:
            config.naming_strategy = "original"

        # 选择输出格式
        print(f"\n{self.lang.t('output_format')}")
        output_formats = [".csv", ".txt", ".xlsx", ".json"]
        for i, fmt in enumerate(output_formats, 1):
            print(f"  {i}. {fmt}")

        try:
            output_choice = int(input(f"\n{self.lang.t('action_selection')}") or "1")
            if 1 <= output_choice <= len(output_formats):
                config.output_extension = output_formats[output_choice - 1]
            else:
                config.output_extension = ".csv"
        except ValueError:
            config.output_extension = ".csv"

        # 批量处理延迟
        try:
            delay = float(input(f"\n{self.lang.t('batch_delay')}") or "2")
            config.batch_delay = delay
        except ValueError:
            config.batch_delay = 2.0

        return config

    def _select_source_folder_with_options(self):
        """选择源文件夹 - 提供自定义路径和演示路径选项"""
        print(f"\n{self.lang.t('select_source_folder_with_options')}")
        print("=" * 40)

        while True:
            print(f"\n{self.lang.t('path_selection_title')}")
            print(f"  {self.lang.t('path_selection_custom')}")
            print(f"  {self.lang.t('path_selection_demo')} ({self.lang.t('demo_source_path')})")

            choice = input(f"\n{self.lang.t('path_selection_prompt')}")

            if choice == "1":
                # 自定义路径
                folder = self._get_folder_path_input(self.lang.t("select_source_folder"))
                if folder:
                    return folder
            elif choice == "2":
                # 演示路径
                demo_path = self.lang.t("demo_source_path")
                folder = os.path.abspath(demo_path)
                print(f"\n{self.lang.t('select_source_folder')} {folder}")
                return self._validate_and_create_folder(folder)
            else:
                print(self.lang.t("invalid_choice"))

    def _configure_output_options_with_options(self):
        """配置输出选项 - 提供自定义路径和演示路径选项"""
        print(f"\n{self.lang.t('output_config')}")
        print("=" * 40)

        while True:
            print(f"\n{self.lang.t('path_selection_title')}")
            print(f"  {self.lang.t('path_selection_custom')}")
            print(f"  {self.lang.t('path_selection_demo')} ({self.lang.t('demo_output_path')})")
            print(f"  3. {self.lang.t('use_source_folder')}")

            choice = input(f"\n{self.lang.t('path_selection_prompt')} (或输入3使用源文件夹): ")

            if choice == "1":
                # 自定义路径
                folder = self._get_folder_path_input(self.lang.t("output_folder_prompt"), allow_empty=False)
                if folder:
                    return folder
            elif choice == "2":
                # 演示路径
                demo_path = self.lang.t("demo_output_path")
                folder = os.path.abspath(demo_path)
                print(f"\n{self.lang.t('output_folder_prompt')} {folder}")
                return self._validate_and_create_folder(folder)
            elif choice == "3":
                # 使用源文件夹
                return ""
            else:
                print(self.lang.t("invalid_choice"))

    def _get_folder_path_input(self, prompt, allow_empty=True):
        """获取文件夹路径输入"""
        while True:
            folder = input(f"\n{prompt}").strip()

            if not folder and allow_empty:
                return ""

            if not folder:
                print(self.lang.t("error") + ": " + self.lang.t("folder_not_exist"))
                continue

            result = self._validate_and_create_folder(folder)
            if result:
                return result

    def _validate_and_create_folder(self, folder):
        """验证并创建文件夹"""
        if not os.path.exists(folder):
            create = input(self.lang.t("create_folder", folder=folder)).lower()
            if create in ["y", "yes", "是", "确认"]:
                try:
                    os.makedirs(folder, exist_ok=True)
                    print(self.lang.t("folder_created", folder=folder))
                    return os.path.abspath(folder)
                except Exception as e:
                    print(f"{self.lang.t('error')}: {str(e)}")
                    return None
            else:
                return None
        elif os.path.isdir(folder):
            return os.path.abspath(folder)
        else:
            print(self.lang.t("not_folder"))
            return None

    def _select_source_folder(self):
        """选择源文件夹（保留原方法以兼容）"""
        while True:
            folder = input(f"\n{self.lang.t('select_source_folder')}").strip()

            if not folder:
                print(f"{self.lang.t('error')}: {self.lang.t('folder_not_exist')}")
                continue

            if not os.path.exists(folder):
                create = input(self.lang.t("create_folder", folder=folder)).lower()
                if create in ["y", "yes", "是", "确认"]:
                    os.makedirs(folder, exist_ok=True)
                    print(self.lang.t("folder_created", folder=folder))
                else:
                    continue

            if os.path.isdir(folder):
                return os.path.abspath(folder)
            else:
                print(self.lang.t("not_folder"))

    def _search_and_select_files(self, folder: str, config: FileProcessingConfig):
        """搜索并选择文件"""
        print(f"\n{self.lang.t('searching_files')}")
        print(self.lang.t("search_folder", folder=folder))
        print(self.lang.t("search_formats", formats=", ".join(config.source_extensions)))
        print(self.lang.t("search_subfolders_status", status=self.lang.t("yes") if config.search_subfolders else self.lang.t("no")))

        try:
            files = FileSearcher.find_files(folder, config.source_extensions, config.search_subfolders)

            if not files:
                print(self.lang.t("file_not_found"))
                return []

            grouped = FileSearcher.group_files_by_extension(files)
            print(self.lang.t("files_found", count=len(files)))

            for ext, file_list in grouped.items():
                print(f"  {ext}: {len(file_list)} {self.lang.t('files')}")

            show_list = input(f"\n{self.lang.t('show_file_list')}").lower()
            if show_list == "y":
                for i, file in enumerate(files, 1):
                    rel_path = os.path.relpath(file, folder)
                    print(f"  {i:3d}. {rel_path}")

            print(f"\n{self.lang.t('select_files')}")
            print(f"  1. {self.lang.t('process_all')}")
            print(f"  2. {self.lang.t('process_specific')}")

            choice = input(f"\n{self.lang.t('action_selection')}") or "1"

            if choice == "1":
                selected_files = files
            elif choice == "2":
                selected_files = self._select_specific_files(files, folder)
            else:
                selected_files = files

            print(self.lang.t("files_selected", count=len(selected_files)))
            return selected_files

        except Exception as e:
            print(self.lang.t("search_error", error=str(e)))
            return []

    def _select_specific_files(self, all_files: List[str], base_folder: str):
        """选择特定文件"""
        print(f"\n{self.lang.t('select_specific_prompt')}")

        for i, file in enumerate(all_files, 1):
            rel_path = os.path.relpath(file, base_folder)
            print(f"  {i:3d}. {rel_path}")

        try:
            selection = input(f"\n{self.lang.t('select_specific_prompt')} (1-{len(all_files)}): ")
            indices = [int(idx.strip()) for idx in selection.split(",")]

            valid_indices = [idx for idx in indices if 1 <= idx <= len(all_files)]

            if not valid_indices:
                print(self.lang.t("input_error") + ", " + self.lang.t("process_all"))
                return all_files

            selected_files = [all_files[idx - 1] for idx in valid_indices]
            return selected_files

        except ValueError:
            print(self.lang.t("input_error") + ", " + self.lang.t("process_all"))
            return all_files

    def _configure_output_options(self):
        """配置输出选项（保留原方法以兼容）"""
        print(f"\n{self.lang.t('output_config')}")
        print("=" * 40)

        while True:
            output_folder = input(f"\n{self.lang.t('output_folder_prompt')}").strip()

            if not output_folder:
                use_source = input(f"{self.lang.t('use_source_folder')}").lower()
                if use_source in ["y", "yes", "是", "确认"]:
                    return ""
                else:
                    continue

            if not os.path.exists(output_folder):
                create = input(self.lang.t("create_folder", folder=output_folder)).lower()
                if create in ["y", "yes", "是", "确认"]:
                    os.makedirs(output_folder, exist_ok=True)
                    print(self.lang.t("folder_created", folder=output_folder))
                    return os.path.abspath(output_folder)
                else:
                    continue

            if os.path.isdir(output_folder):
                return os.path.abspath(output_folder)
            else:
                print(self.lang.t("not_folder"))

    def _preview_and_confirm(self, profile, source_folder, files, file_config):
        """预览并确认导出设置"""
        print(f"\n{self.lang.t('preview_title')}")
        print("=" * 60)

        print(self.lang.t("software_config", name=profile.software_name))
        print(self.lang.t("source_folder", folder=source_folder))
        print(self.lang.t("file_formats", formats=", ".join(file_config.source_extensions)))
        print(self.lang.t("search_subfolders_status", status=self.lang.t("yes") if file_config.search_subfolders else self.lang.t("no")))
        print(self.lang.t("file_count", count=len(files)))
        print(self.lang.t("naming_strategy_status", strategy=file_config.naming_strategy))

        if file_config.output_folder:
            print(self.lang.t("output_folder_status", folder=file_config.output_folder))
        else:
            print(self.lang.t("output_folder_status", folder=self.lang.t("source_folder")))

        print(self.lang.t("output_format_status", format=file_config.output_extension))

        # 显示示例文件名
        if files:
            print(f"\n{self.lang.t('filename_examples')}")
            name_gen = FilenameGenerator(file_config)

            for i in range(min(3, len(files))):
                output_name = name_gen.generate_output_filename(files[i], index=i + 1, total=len(files))
                print(f"  {os.path.basename(files[i])} → {output_name}")

            if len(files) > 3:
                print(self.lang.t("and_more", count=len(files) - 3))

        print("\n" + "=" * 60)

        confirm = input(f"\n{self.lang.t('confirm_export')}").lower()
        return confirm in ["y", "yes", "是", "确认"]

    def _execute_batch_export(self, profile, batch_job: BatchJob):
        """执行批量导出"""
        print(f"\n{self.lang.t('export_starting')}")
        print("=" * 60)

        executor = AutomationExecutor(profile, self.lang)

        # 激活软件窗口
        print(f"\n{self.lang.t('activating_window')}")
        if not executor.activate_window():
            print(f"{self.lang.t('warning')}: {self.lang.t('manual_activate_prompt')}")
            input(f"{self.lang.t('manual_activate_prompt')}")

        # 执行批量导出
        print(f"\n{self.lang.t('processing')}")
        results = executor.execute_batch(batch_job)

        # 更新作业状态并显示结果
        batch_job.status = "completed"
        batch_job.end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        batch_job.processed_count = results["success"]
        batch_job.failed_count = results["failed"]

        print(f"\n{self.lang.t('export_completed')}")
        print("=" * 60)

        print(f"\n{self.lang.t('statistics')}")
        print(f"  {self.lang.t('total_files', count=results['total'])}")
        print(f"  {self.lang.t('success_count', count=results['success'])}")
        print(f"  {self.lang.t('failed_count', count=results['failed'])}")

        if results["total"] > 0:
            success_rate = results["success"] / results["total"] * 100
            print(f"  {self.lang.t('success_rate', rate=success_rate)}")

        if results["failed"] > 0:
            print(f"\n{self.lang.t('failed_files')}")
            for file in results["failed_files"][:10]:
                print(f"   - {os.path.basename(file)}")
            if len(results["failed_files"]) > 10:
                print(self.lang.t("and_more", count=len(results["failed_files"]) - 10))

        print(f"\n{self.lang.t('start_time', time=batch_job.start_time)}")
        print(f"  {self.lang.t('end_time', time=batch_job.end_time)}")

        # 询问是否保存结果报告
        save_report = input(f"\n{self.lang.t('save_report')}").lower()
        if save_report in ["y", "yes", "是", "确认"]:
            self._save_export_report(batch_job, results)

    def _save_export_report(self, job: BatchJob, results: Dict):
        """保存导出报告"""
        report_dir = "export_reports"
        os.makedirs(report_dir, exist_ok=True)

        report_file = os.path.join(report_dir, f"export_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

        with open(report_file, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write(f"{self.lang.t('export_report')}\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"{self.lang.t('software_profile')}: {job.profile_name}\n")
            f.write(f"{self.lang.t('source_folder')}: {job.source_folder}\n")
            f.write(f"{self.lang.t('processing_time')}: {job.start_time} - {job.end_time}\n\n")

            f.write(f"{self.lang.t('statistics')}:\n")
            f.write(f"  {self.lang.t('total_files', count=results['total'])}\n")
            f.write(f"  {self.lang.t('success_count', count=results['success'])}\n")
            f.write(f"  {self.lang.t('failed_count', count=results['failed'])}\n\n")

            if results["output_files"]:
                f.write(f"{self.lang.t('successful_exports')}:\n")
                for file in results["output_files"]:
                    f.write(f"  ✓ {file}\n")

            if results["failed_files"]:
                f.write(f"\n{self.lang.t('failed_exports')}:\n")
                for file in results["failed_files"]:
                    f.write(f"  ✗ {file}\n")

        print(self.lang.t("report_saved", file=report_file))

    def view_export_history(self):
        """查看导出历史"""
        history = self.config_manager.load_export_history()

        if not history:
            print(f"\n{self.lang.t('no_history')}")
            return

        print(f"\n{self.lang.t('history_title', count=len(history))}")
        print("=" * 60)

        for i, job in enumerate(reversed(history[-10:]), 1):
            print(f"\n{i}. {self.lang.t('job_id', id=job.get('job_id', 'N/A'))}")
            print(f"   {self.lang.t('job_software', name=job.get('profile_name', 'N/A'))}")
            print(f"   {self.lang.t('job_source', folder=job.get('source_folder', 'N/A'))}")
            print(f"   {self.lang.t('job_file_count', count=len(job.get('files_to_process', [])))}")
            print(f"   {self.lang.t('job_status', status=job.get('status', 'N/A'))}")
            print(f"   {self.lang.t('job_start_time', time=job.get('start_time', 'N/A'))}")

            if job.get("processed_count", 0) > 0:
                success = job.get("processed_count", 0)
                total = len(job.get("files_to_process", []))
                print(f"   {self.lang.t('success_count', count=success)}/{total}")

        print("\n" + "=" * 60)

    def manage_profiles(self):
        """管理配置档案"""
        profiles = self.config_manager.list_profiles()
        if not profiles:
            print(f"\n{self.lang.t('no_profiles')}")
            return

        print(f"\n{self.lang.t('profile_list')}")
        for i, name in enumerate(profiles, 1):
            profile = self.config_manager.load_profile(name)
            print(f"{i}. {name}")
            print(f"   {self.lang.t('profile_desc', desc=profile.description)}")
            print(f"   {self.lang.t('profile_created', time=profile.created_time)}")
            print(f"   {self.lang.t('profile_formats', formats=', '.join(profile.file_extensions))}")
            print()

        print(f"\n{self.lang.t('manage_options')}")
        print(f"1. {self.lang.t('delete_profile')}")
        print(f"2. {self.lang.t('back')}")

        choice = input(f"\n{self.lang.t('action_selection')}")
        if choice == "1":
            try:
                num = int(input(f"{self.lang.t('select_delete')}")) - 1
                if 0 <= num < len(profiles):
                    confirm = input(self.lang.t("confirm_delete", name=profiles[num])).lower()
                    if confirm in ["y", "yes", "是", "确认"]:
                        if self.config_manager.delete_profile(profiles[num]):
                            print(self.lang.t("delete_success"))
                        else:
                            print(self.lang.t("delete_failed"))
                else:
                    print(self.lang.t("invalid_choice"))
            except ValueError:
                print(self.lang.t("input_error"))

    def edit_profile(self):
        """编辑现有配置"""
        profiles = self.config_manager.list_profiles()
        if not profiles:
            print(f"\n{self.lang.t('no_profiles')}")
            return

        print(f"\n{self.lang.t('select_profile')}:")
        for i, name in enumerate(profiles, 1):
            print(f"{i}. {name}")

        try:
            choice = int(input(f"\n{self.lang.t('select_delete')}")) - 1
            if 0 <= choice < len(profiles):
                profile_name = profiles[choice]
                profile = self.config_manager.load_profile(profile_name)

                print(f"\n{self.lang.t('edit_profile', name=profile.software_name)}")
                edit_options = self.lang.t("edit_options").split(",")
                for i, option in enumerate(edit_options, 1):
                    print(f"  {i}. {option.strip()}")

                edit_choice = input(f"\n{self.lang.t('action_selection')}")

                if edit_choice == "1":
                    self._edit_basic_info(profile)
                elif edit_choice == "2":
                    print(f"\n{self.lang.t('rerun_actions')}")
                    new_actions = self.setup_wizard._record_actions_interactive()
                    if new_actions:
                        profile.actions = new_actions
                        self.config_manager.save_profile(profile)
                        print(self.lang.t("actions_updated"))

        except ValueError:
            print(self.lang.t("input_error"))

    def _edit_basic_info(self, profile: SoftwareProfile):
        """编辑基本信息"""
        print(f"\n{self.lang.t('edit_basic_info')}")

        new_name = input(self.lang.t("edit_software_name", current=profile.software_name)) or profile.software_name
        new_desc = input(self.lang.t("edit_description", current=profile.description)) or profile.description
        new_title = input(self.lang.t("edit_window_title", current=profile.window_title)) or profile.window_title

        profile.software_name = new_name
        profile.description = new_desc
        profile.window_title = new_title

        if self.config_manager.save_profile(profile):
            print(self.lang.t("update_success"))
        else:
            print(self.lang.t("update_failed"))


def run_auto_export(input_path: str, out_dir: str, **kwargs) -> Dict[str, str]:
    """
    Run auto export in non-interactive mode.
    This function integrates the interactive tool into the framework.

    Args:
        input_path: Path to input data or directory
        out_dir: Output directory
        **kwargs: Additional parameters (e.g., profile_name, file_formats, etc.)

    Returns:
        Dict with status and output info
    """
    try:
        # Initialize components
        lang_manager = LanguageManager()
        config_manager = ConfigManager()

        # Get parameters from kwargs or use defaults
        profile_name = kwargs.get("profile_name")
        file_formats = kwargs.get("file_formats", [".xlsx", ".xls"])
        search_subfolders = kwargs.get("search_subfolders", True)
        naming_strategy = kwargs.get("naming_strategy", "original")
        output_format = kwargs.get("output_format", "csv")

        if not profile_name:
            return {"status": "error", "message": "profile_name is required in kwargs"}

        # Load profile
        profile = config_manager.load_profile(profile_name)
        if not profile:
            return {"status": "error", "message": f"Profile '{profile_name}' not found"}

        # Configure file processing
        file_config = FileProcessingConfig(
            source_extensions=file_formats,
            search_subfolders=search_subfolders,
            naming_strategy=naming_strategy,
            output_extension=f".{output_format}",
            output_folder=out_dir,
            batch_delay=kwargs.get("batch_delay", 2.0),
        )

        # Determine source folder
        if os.path.isdir(input_path):
            source_folder = input_path
        else:
            source_folder = os.path.dirname(input_path)

        # Search files
        all_files = FileSearcher.find_files(source_folder, file_config.source_extensions, file_config.search_subfolders)
        if not all_files:
            return {"status": "error", "message": f"No files found in {source_folder} with formats {file_config.source_extensions}"}

        # Create batch job
        batch_job = BatchJob(profile_name=profile_name, source_folder=source_folder, file_config=file_config, files_to_process=all_files, start_time=datetime.now())

        # Execute batch export
        executor = AutomationExecutor(profile, lang_manager)
        results = executor.execute_batch(batch_job)

        # Save results
        config_manager.save_export_history(batch_job)

        return {
            "status": "success",
            "output_path": out_dir,
            "processed_files": len(results.get("output_files", [])),
            "failed_files": len(results.get("failed_files", [])),
            "message": "Auto export completed",
        }

    except Exception as e:
        return {"status": "error", "message": f"Auto export failed: {str(e)}"}


# ==================== 程序入口 ====================
def main():
    try:
        # 创建助手实例
        assistant = MultilingualExportAssistant()

        # 选择语言
        assistant.select_language()

        # 显示主菜单
        assistant.main_menu()

    except KeyboardInterrupt:
        print(f"\n\n{assistant.lang.t('program_interrupted')}")
    except Exception as e:
        print(f"\n{assistant.lang.t('program_error', error=str(e))}")
        import traceback

        traceback.print_exc()
    finally:
        print(f"\n{assistant.lang.t('exit')}")


if __name__ == "__main__":
    main()
