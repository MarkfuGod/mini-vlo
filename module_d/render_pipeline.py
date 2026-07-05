import bpy
import os
import math
import json
# ================= 配置区 =================
# 1. 你的动作数据文件夹 (放 .fbx 或 .bvh 文件)
MOTION_DATA_PATH = "C:/Users/lcy/Desktop/HKU final project/motions/" 
# 2. 视频输出的总文件夹
OUTPUT_BASE_DIR = "C:/Users/lcy/Desktop/HKU final project/output_videos/"
# 3. 渲染引擎选择: 'BLENDER_EEVEE'
RENDER_ENGINE = 'BLENDER_EEVEE'

os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

def setup_render_settings():
    """设置视频输出格式和引擎 (完美适配 Blender 5.1)"""
    scene = bpy.context.scene
    scene.render.engine = RENDER_ENGINE
    
    # === 核心修改点：应用你抓取到的 5.1 版本 API ===
    scene.render.image_settings.file_format = 'FFMPEG'
    
    # 兼容 5.1 的新媒体类型属性
    try:
        scene.render.image_settings.media_type = 'VIDEO'
    except AttributeError:
        pass 

    scene.render.ffmpeg.format = 'MPEG4'
    scene.render.ffmpeg.codec = 'H264'
    scene.render.ffmpeg.constant_rate_factor = 'MEDIUM'
    
    # 分辨率
    #scene.render.resolution_x = 512
    #scene.render.resolution_y = 512
    # 1. 提高分辨率 (例如改为 1024x1024 或 1920x1080)
    scene.render.resolution_x = 1024 
    scene.render.resolution_y = 1024
    
    # 2. 提升画面百分比 (确保是 100%)
    scene.render.resolution_percentage = 100

    # 3. 提高 EEVEE 引擎的抗锯齿采样率 (让边缘更平滑，画质更清晰)
    scene.eevee.taa_render_samples = 64

def clear_imported_objects():
    """清理上一轮导入的骨架和物体，保持场景上下文"""
    for obj in bpy.data.objects:
        # === 修改点 1：把 '苏珊娜' 加入不删除的白名单 ===
        if obj.type in ['ARMATURE', 'MESH'] and obj.name not in ['平面', '苏珊娜']: 
            bpy.data.objects.remove(obj, do_unlink=True)

def add_scene_context(action_name):
    """
    根据动作名称中的关键字，动态生成简单的场景几何体
    """
    name_lower = action_name.lower()
    
        
    # 情况 1：跨越、跳跃动作 -> 生成长方体(障碍物/箱子)
    if "jump" in name_lower or "vault" in name_lower or "step" in name_lower:
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, -0.020894, 0.608734))
        obj = bpy.context.active_object
        obj.name = "Context_Obstacle"
        # 调整缩放使其看起来像一个横杆或矮墙: X轴拉长，Y轴变窄，Z轴压扁
        obj.scale = (2.0, 0.3, 0.5) 
        
    # 情况 2：坐下动作 -> 生成方块(椅子)
    elif "sit" in name_lower:
        bpy.ops.mesh.primitive_cube_add(size=0.5, location=(0, 0.5, 0.25))
        obj = bpy.context.active_object
        obj.name = "Context_Chair"
        
    print(f"  -> 已为动作 '{action_name}' 匹配并生成场景上下文。")
    
    
def export_trajectory_to_json(armature, scene, video_folder, file_name):
    """
    遍历时间轴，提取骨骼 3D 轨迹并保存为 JSON
    """
    trajectory_data = {}
    
    # 定义你要提取的部位，以及它们在不同动捕标准下可能出现的名字
    target_bones_keywords = {
        "Root": ["Hips", "hips", "Root", "root", "Pelvis", "pelvis"],
        "Hand_R": ["RightHand", "Hand_R", "hand.R", "hand_r"],
        "Hand_L": ["LeftHand", "Hand_L", "hand.L", "hand_l"]
    }
    
    # 动态匹配当前骨架里实际的骨骼名称
    actual_bones = {}
    for target_key, keywords in target_bones_keywords.items():
        for pb in armature.pose.bones:
            if any(keyword in pb.name for keyword in keywords):
                actual_bones[target_key] = pb.name
                break
                
    # 如果找不到根节点，直接返回
    if "Root" not in actual_bones:
        print(f"⚠️ 未能在 {file_name} 中找到 Root 骨骼，跳过轨迹导出。")
        return

    print(f"开始提取 {file_name} 的 3D 轨迹数据...")
    
    # 遍历动画的每一帧
    for frame in range(scene.frame_start, scene.frame_end + 1):
        scene.frame_set(frame)
        # === 核心：必须强制刷新视图层，否则骨骼矩阵不会随动画更新 ===
        bpy.context.view_layer.update() 
        
        frame_data = {}
        for target_key, bone_name in actual_bones.items():
            pose_bone = armature.pose.bones[bone_name]
            # 计算骨骼在世界坐标系下的真实物理位置 (骨架矩阵 @ 骨骼矩阵)
            world_loc = armature.matrix_world @ pose_bone.matrix.translation
            
            # 保留 4 位小数，使 JSON 文件不会过于庞大
            frame_data[target_key] = {
                "x": round(world_loc.x, 4), 
                "y": round(world_loc.y, 4), 
                "z": round(world_loc.z, 4)
            }
            
        trajectory_data[f"frame_{frame}"] = frame_data

    # 写入 JSON 文件到对应的视频文件夹下
    json_path = os.path.join(video_folder, f"{file_name}_trajectory.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(trajectory_data, f, indent=4)
        
    print(f"✅ 已成功导出 3D 轨迹至: {json_path}")
    
    
def render_action_video(action_file):
    """处理单个动作文件的渲染逻辑"""
    file_name = os.path.basename(action_file).split('.')[0]
    video_folder = os.path.join(OUTPUT_BASE_DIR, file_name)
    os.makedirs(video_folder, exist_ok=True)

    # 1. 导入动作 (加入容错机制)
    try:
        if action_file.endswith('.fbx'):
            bpy.ops.import_scene.fbx(filepath=action_file)
        elif action_file.endswith('.bvh'):
            bpy.ops.import_anim.bvh(filepath=action_file)
        # ================= 新增：自动生成场景上下文 =================
        add_scene_context(file_name)
    except RuntimeError as e:
        print(f"⚠️ 跳过文件 {action_file}: 导入失败，可能文件已损坏。错误信息: {e}")
        return # 结束当前有问题文件的处理，直接进入下一个循环

# 2. 自动设置动画时长
    scene = bpy.context.scene
    armature = next((o for o in bpy.data.objects if o.type == 'ARMATURE'), None)
    if armature and armature.animation_data:
        action = armature.animation_data.action
        scene.frame_start = int(action.frame_range[0])
        scene.frame_end = int(action.frame_range[1])
        
        # ================= 新增：提取并导出 3D 轨迹 =================
        export_trajectory_to_json(armature, scene, video_folder, file_name)
        # ==========================================================

    # ================= 自动化绑定第一人称相机 =================
    cam_ego = bpy.data.objects.get("摄像机")
    # ... 后续代码保持不变 ...

    if cam_ego and armature:
        head_bone_name = None
        for bone in armature.data.bones:
            if "Head" in bone.name or "head" in bone.name:
                head_bone_name = bone.name
                break
        
        if head_bone_name:
            cam_ego.parent = armature
            cam_ego.parent_type = 'BONE'
            cam_ego.parent_bone = head_bone_name
            
            cam_ego.location = (0, 0, 2) 
            
            rot_x = math.radians(180)  
            rot_y = math.radians(0)    
            rot_z = math.radians(180)  
            
            cam_ego.rotation_euler = (rot_x, rot_y, rot_z) 
            
            print(f"成功将摄像机绑定到新骨架的 {head_bone_name} 骨骼上！视角已修正。")

    # 3. 渲染多视角 
    cameras = {
        "Fixed_View": bpy.data.objects.get("Camera"),
        "Ego_View": bpy.data.objects.get("摄像机")
    }

    for view_name, cam in cameras.items():
        if cam:
            scene.camera = cam
            # === 核心修改点：改回 .mp4 后缀 ===
            scene.render.filepath = os.path.join(video_folder, f"{view_name}.mp4")
            print(f"正在渲染 {file_name} 的 {view_name} 视角...")
            bpy.ops.render.render(animation=True)

# ================= 主执行逻辑 =================
setup_render_settings()

for file in os.listdir(MOTION_DATA_PATH):
    if file.endswith(('.fbx', '.bvh')):
        clear_imported_objects() 
        render_action_video(os.path.join(MOTION_DATA_PATH, file))
        
    

print("所有动作视频渲染任务已完成！")