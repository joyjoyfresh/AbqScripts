#!/bin/sh
set -u  # 使用未定义变量时报错

TAG='cqu-login'  # 定义日志标签
CONF_PATH='/etc/cqu-login.conf'  # 定义配置文件路径
HOTPLUG_PATH='/etc/hotplug.d/iface/99-cqu-login'  # 定义热插拔脚本路径
WATCHDOG_PATH='/usr/bin/cqu-login-watchdog.sh'  # 定义巡检脚本路径
SELF_INSTALL_PATH='/usr/bin/cqu-login.sh'  # 定义主脚本安装路径
LOGIN_LOCK_FILE='/tmp/cqu-login-main.lock'  # 定义全局登录流程锁文件路径

write_default_conf() {  # 定义写入默认配置函数
	umask 077  # 设置严格文件权限掩码
	cat > "${CONF_PATH}" <<'EOF'  # 写入默认配置文件
CQU_ACCOUNT=',0,202416021194T'  # 设置校园网账号并保留,0,前缀
CQU_PASSWORD='FYXfyx20010805'  # 设置校园网密码占位符
CQU_IFACE='wan'  # 设置WAN逻辑接口名
CQU_MAC_IF='eth0'  # 设置用于读取MAC的接口名
CQU_PING_A='223.5.5.5'  # 设置连通性检测地址1
CQU_PING_B='1.1.1.1'  # 设置连通性检测地址2
EOF
	chmod 600 "${CONF_PATH}"  # 设置配置文件仅root可读写
}  # 结束写入默认配置函数

load_conf() {  # 定义加载配置函数
	[ -f "${CONF_PATH}" ] || write_default_conf  # 配置不存在时写入默认配置
	. "${CONF_PATH}"  # 加载配置变量
}  # 结束加载配置函数

get_wan_ip() {  # 定义获取WAN地址函数
	ubus call network.interface."${CQU_IFACE}" status 2>/dev/null | jsonfilter -e '@["ipv4-address"][0].address'  # 从ubus提取IPv4地址
}  # 结束获取WAN地址函数

get_mac_no_colon() {  # 定义获取无冒号MAC函数
	local raw  # 定义局部变量保存原始MAC
	raw="$(cat /sys/class/net/"${CQU_MAC_IF}"/address 2>/dev/null)" || return 1  # 读取网卡MAC地址
	[ -n "${raw}" ] || return 1  # 校验MAC非空
	echo "${raw}" | tr 'A-F' 'a-f' | tr -d ':'  # 转为小写并去掉冒号
}  # 结束获取无冒号MAC函数

login_once() {  # 定义单次登录函数
	local wan_ip  # 定义局部变量保存WAN地址
	local mac  # 定义局部变量保存MAC
	local now  # 定义局部变量保存时间戳
	local resp  # 定义局部变量保存请求响应

	load_conf  # 加载配置
	wan_ip="$(get_wan_ip)"  # 获取当前WAN地址
	[ -n "${wan_ip}" ] || { logger -t "${TAG}" "未获取到WAN地址，接口=${CQU_IFACE}"; return 2; }  # WAN地址为空时返回错误

	mac="$(get_mac_no_colon)"  # 获取无冒号MAC
	[ -n "${mac}" ] || { logger -t "${TAG}" "未获取到MAC地址，接口=${CQU_MAC_IF}"; return 3; }  # MAC为空时返回错误

	now="$(date +%s)"  # 生成秒级时间戳参数

	resp="$(curl -sS --get 'http://login.cqu.edu.cn:801/eportal/portal/login' \
	  --connect-timeout 5 \
	  --max-time 12 \
	  --data-urlencode 'callback=dr1004' \
	  --data-urlencode 'login_method=1' \
	  --data-urlencode "user_account=${CQU_ACCOUNT}" \
	  --data-urlencode "user_password=${CQU_PASSWORD}" \
	  --data-urlencode "wlan_user_ip=${wan_ip}" \
	  --data-urlencode 'wlan_user_ipv6=' \
	  --data-urlencode "wlan_user_mac=${mac}" \
	  --data-urlencode 'wlan_ac_ip=' \
	  --data-urlencode 'wlan_ac_name=' \
	  --data-urlencode 'term_ua=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36 Edg/147.0.0.0' \
	  --data-urlencode 'term_type=1' \
	  --data-urlencode 'jsVersion=4.2.2' \
	  --data-urlencode 'terminal_type=1' \
	  --data-urlencode 'lang=zh-cn' \
	  --data-urlencode "v=${now}" \
	  --data-urlencode 'lang=zh' \
	  -H 'Accept: */*' \
	  -H 'Referer: http://login.cqu.edu.cn/' \
	  -H 'Connection: keep-alive' \
	  --insecure 2>&1)"  # 发起登录请求并捕获输出

	echo "${resp}" | grep -Eqi '"result"[[:space:]]*:[[:space:]]*"?1"?|"ret_code"[[:space:]]*:[[:space:]]*2|success|成功|已在线|已经在线|already' && { logger -t "${TAG}" "登录成功或已在线，wan_ip=${wan_ip}"; return 0; }  # 识别成功或已在线

	logger -t "${TAG}" "登录失败，响应=${resp}"  # 记录失败响应
	return 4  # 返回登录失败状态码
}  # 结束单次登录函数

login_retry() {  # 定义重试登录函数
	local i=1  # 初始化重试计数器
	while [ "${i}" -le 3 ]; do  # 最多尝试3次
		login_once && return 0  # 登录成功即返回
		logger -t "${TAG}" "第${i}次登录失败，准备重试"  # 记录失败次数
		i=$((i + 1))  # 计数器加一
		sleep 2  # 两秒后重试
	done
	logger -t "${TAG}" '连续3次登录失败'  # 记录最终失败
	return 5  # 返回最终失败状态码
}  # 结束重试登录函数

acquire_login_lock() {  # 定义获取全局登录锁函数
	local old_pid  # 定义局部变量保存旧进程号
	old_pid="$(cat "${LOGIN_LOCK_FILE}" 2>/dev/null)"  # 读取旧锁对应进程号
	[ -n "${old_pid}" ] && kill -0 "${old_pid}" 2>/dev/null && return 1  # 若旧进程仍存活则返回失败
	echo "$$" > "${LOGIN_LOCK_FILE}"  # 写入当前进程号作为新锁
	return 0  # 返回获取锁成功
}  # 结束获取全局登录锁函数

release_login_lock() {  # 定义释放全局登录锁函数
	local lock_pid  # 定义局部变量保存锁中的进程号
	lock_pid="$(cat "${LOGIN_LOCK_FILE}" 2>/dev/null)"  # 读取当前锁进程号
	[ "${lock_pid}" = "$$" ] && rm -f "${LOGIN_LOCK_FILE}"  # 仅删除属于当前进程的锁文件
	return 0  # 返回释放锁完成
}  # 结束释放全局登录锁函数

run_login_with_lock() {  # 定义带锁的登录执行函数
	acquire_login_lock || { logger -t "${TAG}" '已有登录流程在执行，跳过本次'; return 0; }  # 获取锁失败时跳过本次执行
	login_retry  # 获取锁后执行登录重试
	LOGIN_RC="$?"  # 保存登录返回码
	release_login_lock  # 执行完成后释放全局登录锁
	return "${LOGIN_RC}"  # 返回登录执行结果
}  # 结束带锁的登录执行函数

write_watchdog() {  # 定义写入巡检脚本函数
	cat > "${WATCHDOG_PATH}" <<'EOF'  # 写入巡检脚本文件
#!/bin/sh
set -u  # 使用未定义变量时报错

TAG='cqu-watchdog'  # 定义巡检日志标签
CONF_PATH='/etc/cqu-login.conf'  # 定义配置文件路径

[ -f "${CONF_PATH}" ] || exit 1  # 配置不存在时退出
. "${CONF_PATH}"  # 加载配置变量

ping -c 1 -W 2 "${CQU_PING_A}" >/dev/null 2>&1 && exit 0  # 地址1可达时退出
ping -c 1 -W 2 "${CQU_PING_B}" >/dev/null 2>&1 && exit 0  # 地址2可达时退出

logger -t "${TAG}" '检测到网络不可达，开始重登'  # 记录重登触发
/usr/bin/cqu-login.sh login >/dev/null 2>&1  # 调用主脚本执行登录
exit 0  # 正常结束巡检脚本
EOF
	chmod +x "${WATCHDOG_PATH}"  # 赋予巡检脚本可执行权限
}  # 结束写入巡检脚本函数

write_hotplug() {  # 定义写入热插拔脚本函数
	cat > "${HOTPLUG_PATH}" <<'EOF'  # 写入hotplug触发脚本
#!/bin/sh
[ "${ACTION}" = "ifup" ] || exit 0  # 仅在ifup事件触发
[ "${INTERFACE}" = "wan" ] || exit 0  # 仅处理wan接口
/usr/bin/cqu-login.sh login >/dev/null 2>&1 &  # 后台执行登录避免阻塞
LOCK_FILE='/tmp/cqu-login-bootquick.lock'  # 定义开机快速巡检锁文件
(  # 启动后台快速巡检子任务
	OLD_PID="$(cat "${LOCK_FILE}" 2>/dev/null)"  # 读取旧的巡检进程号
	[ -n "${OLD_PID}" ] && kill -0 "${OLD_PID}" 2>/dev/null && exit 0  # 已有巡检进程在运行则直接退出
	echo "$$" > "${LOCK_FILE}"  # 写入当前进程号到锁文件
	i=1  # 初始化快速巡检计数器
	while [ "${i}" -le 6 ]; do  # 前2分钟每20秒巡检一次共6轮
		/usr/bin/cqu-login.sh watchdog >/dev/null 2>&1  # 执行一次巡检并在离线时重登
		i=$((i + 1))  # 巡检计数加一
		sleep 20  # 每轮间隔20秒
	done
	rm -f "${LOCK_FILE}"  # 快速巡检结束后清理锁文件
) >/dev/null 2>&1 &  # 后台执行快速巡检任务
EOF
	chmod +x "${HOTPLUG_PATH}"  # 赋予hotplug脚本可执行权限
}  # 结束写入热插拔脚本函数

install_self() {  # 定义安装主脚本函数
	cp "$0" "${SELF_INSTALL_PATH}"  # 复制当前脚本到系统路径
	chmod +x "${SELF_INSTALL_PATH}"  # 赋予系统脚本可执行权限
}  # 结束安装主脚本函数

install_cron() {  # 定义安装定时任务函数
	grep -q 'cqu-login-watchdog.sh' /etc/crontabs/root 2>/dev/null || echo '*/5 * * * * /usr/bin/cqu-login-watchdog.sh # cqu-login-watchdog' >> /etc/crontabs/root  # 避免重复追加定时任务
	/etc/init.d/cron restart >/dev/null 2>&1  # 重启cron服务使配置生效
}  # 结束安装定时任务函数

install_all() {  # 定义一键安装函数
	write_default_conf  # 写入默认配置文件
	install_self  # 安装主脚本到系统路径
	write_watchdog  # 写入巡检脚本
	write_hotplug  # 写入热插拔触发脚本
	install_cron  # 安装定时巡检任务
	logger -t "${TAG}" '安装完成，请编辑 /etc/cqu-login.conf 填写新密码'  # 记录安装提示
	echo '安装完成：请先编辑 /etc/cqu-login.conf 的 CQU_PASSWORD，再执行 /usr/bin/cqu-login.sh login'  # 输出安装提示
}  # 结束一键安装函数

show_help() {  # 定义帮助信息函数
	echo '用法: sh autologin.sh [install|login|watchdog|help]'  # 输出用法说明
	echo 'install  : 安装配置、主脚本、hotplug、cron'  # 输出install说明
	echo 'login    : 立即执行一次登录（含重试）'  # 输出login说明
	echo 'watchdog : 执行一次连通性检测，不通则重登'  # 输出watchdog说明
	echo 'help     : 显示帮助'  # 输出help说明
}  # 结束帮助信息函数

run_watchdog_once() {  # 定义单次巡检函数
	load_conf  # 加载配置变量
	ping -c 1 -W 2 "${CQU_PING_A}" >/dev/null 2>&1 && return 0  # 地址1可达则返回成功
	ping -c 1 -W 2 "${CQU_PING_B}" >/dev/null 2>&1 && return 0  # 地址2可达则返回成功
	acquire_login_lock || { logger -t "${TAG}" '已有登录流程在执行，跳过本次巡检重登'; return 0; }  # 获取锁失败时跳过本次重登
	openclash_stop_for_login  # 离线时先尝试关闭OpenClash
	login_retry  # 网络不通时执行登录重试
	LOGIN_RC="$?"  # 保存登录返回码
	openclash_restore_after_login "${LOGIN_RC}"  # 仅在登录成功时恢复OpenClash
	release_login_lock  # 巡检重登流程结束后释放全局登录锁
	return "${LOGIN_RC}"  # 返回登录结果
}  # 结束单次巡检函数

openclash_stop_for_login() {  # 定义登录前关闭OpenClash函数
	OPENCLASH_SHOULD_RESTORE='0'  # 默认不需要恢复OpenClash
	[ -x /etc/init.d/openclash ] || return 0  # 未安装OpenClash时直接返回
	/etc/init.d/openclash stop >/dev/null 2>&1  # 尝试停止OpenClash服务
	OPENCLASH_SHOULD_RESTORE='1'  # 标记登录后需要恢复OpenClash
	logger -t "${TAG}" '检测到离线，已执行OpenClash停止流程'  # 记录关闭OpenClash日志
	return 0  # 返回成功
}  # 结束登录前关闭OpenClash函数

openclash_restore_after_login() {  # 定义登录后恢复OpenClash函数
	LOGIN_RC_FOR_RESTORE="${1:-1}"  # 读取登录返回码并默认失败
	[ "${OPENCLASH_SHOULD_RESTORE:-0}" = '1' ] || return 0  # 无需恢复时直接返回
	[ "${LOGIN_RC_FOR_RESTORE}" -eq 0 ] || { logger -t "${TAG}" '登录失败，保持OpenClash关闭状态'; return 0; }  # 登录失败时不启动OpenClash
	[ -x /etc/init.d/openclash ] || return 0  # OpenClash脚本不存在时直接返回
	/etc/init.d/openclash start >/dev/null 2>&1  # 尝试启动OpenClash服务
	logger -t "${TAG}" '登录流程结束，已执行OpenClash启动流程'  # 记录恢复OpenClash日志
	return 0  # 返回成功
}  # 结束登录后恢复OpenClash函数

CMD="${1:-help}"  # 获取命令参数并设置默认值

case "${CMD}" in  # 根据命令分发执行逻辑
	install) install_all ;;  # 执行一键安装
	login) run_login_with_lock ;;  # 执行带锁登录重试
	watchdog) run_watchdog_once ;;  # 执行单次巡检
	help|*) show_help ;;  # 显示帮助信息
esac
